"""Pruebas de sanidad del entorno de inversión.

Cada grupo de pruebas verifica una propiedad que el entorno debe cumplir:

- Valuación del portafolio: permanecer siempre invertido reproduce comprar y
  mantener, y permanecer siempre en efectivo deja el capital intacto.
- Costos: el cálculo se contrastó con un ejemplo de cinco pasos resuelto a mano.
- Mecánica del episodio: finalización con truncated, reinicio, arranque aleatorio
  reproducible y manejo de acciones inválidas.
- Recompensa intercambiable: el entorno entrega el diccionario info y reinicia la
  recompensa en cada episodio.
- Lookahead: al modificar el futuro, lo que ya ocurrió no cambia.
- Compatibilidad con Gymnasium y Stable-Baselines3.
- Integración con los datos reales: ausencia de NaN, orden de las particiones y
  escalador ajustado solo con entrenamiento.
"""

import numpy as np
import pytest

from src.environment import NetReturnReward, TradingEnvironment

CAPITAL = 100_000.0


def make_env(prices, cost_bps=10.0, features=None, **kwargs):
    prices = np.asarray(prices, dtype=float)
    if features is None:
        # Se usan dos variables sencillas que dependen solo del pasado: el retorno
        # de la vela y un contador, con la intención de probar la mecánica del
        # entorno.
        returns = np.concatenate([[0.0], prices[1:] / prices[:-1] - 1.0])
        features = np.column_stack([returns, np.arange(len(prices)) / len(prices)])
    return TradingEnvironment(
        prices=prices,
        features=features,
        initial_capital=CAPITAL,
        transaction_cost_bps=cost_bps,
        **kwargs,
    )


def run_episode(env, actions):
    """Ejecuta una lista de acciones y devuelve el info de cada paso."""
    env.reset(seed=0)
    infos = []
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        infos.append(info)
        if terminated or truncated:
            break
    return infos


def random_walk(n, seed=1):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))


# ----------------------------------------------------------------------
# Valuación del portafolio
# ----------------------------------------------------------------------
def test_siempre_invertido_replica_comprar_y_mantener():
    prices = random_walk(200)
    env = make_env(prices, cost_bps=0.0)
    infos = run_episode(env, [2] * (len(prices) - 1))
    esperado = CAPITAL * prices[-1] / prices[0]
    assert infos[-1]["portfolio_value"] == pytest.approx(esperado, rel=1e-12)


def test_comprar_y_mantener_con_costo_paga_solo_la_entrada():
    prices = random_walk(200)
    env = make_env(prices, cost_bps=10.0)
    infos = run_episode(env, [2] * (len(prices) - 1))
    esperado = CAPITAL * (1 - 0.001) * prices[-1] / prices[0]
    assert infos[-1]["portfolio_value"] == pytest.approx(esperado, rel=1e-12)
    assert infos[-1]["n_trades"] == 1
    assert infos[-1]["total_cost"] == pytest.approx(CAPITAL * 0.001, rel=1e-12)


def test_siempre_en_efectivo_no_cambia_el_capital_ni_paga_costos():
    prices = random_walk(200)
    env = make_env(prices, cost_bps=50.0)
    infos = run_episode(env, [0] * (len(prices) - 1))
    assert infos[-1]["portfolio_value"] == CAPITAL
    assert infos[-1]["total_cost"] == 0.0
    assert infos[-1]["n_trades"] == 0


def test_posicion_intermedia_expone_la_mitad_del_movimiento():
    env = make_env([100.0, 110.0], cost_bps=0.0)
    infos = run_episode(env, [1])
    assert infos[0]["portfolio_value"] == pytest.approx(CAPITAL * 1.05, rel=1e-12)


# ----------------------------------------------------------------------
# Costos: ejemplo de cinco pasos calculado a mano
# ----------------------------------------------------------------------
def test_costos_cinco_pasos_calculados_a_mano():
    # Retornos de cada paso: +10 %, -10 %, 0 %, +10 % y +10 %.
    prices = [100.0, 110.0, 99.0, 99.0, 108.9, 119.79]
    # Acciones elegidas: 100 %, 50 %, 0 %, 100 % y 100 %.
    infos = run_episode(make_env(prices), [2, 1, 0, 2, 2])

    costos = [100.0, 54.945, 52.171651125, 104.291130598875, 0.0]
    valores = [
        109890.0,
        104343.30225,
        104291.130598875,
        114605.52341510374,
        126066.07575661413,
    ]
    for info, costo, valor in zip(infos, costos, valores):
        assert info["transaction_cost"] == pytest.approx(costo, rel=1e-9, abs=1e-9)
        assert info["portfolio_value"] == pytest.approx(valor, rel=1e-9)
    assert infos[-1]["total_cost"] == pytest.approx(311.407781723875, rel=1e-9)
    assert infos[-1]["n_trades"] == 4


def test_la_recompensa_v1_es_el_retorno_neto_de_costos():
    prices = [100.0, 110.0]
    env = make_env(prices, cost_bps=10.0)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(2)
    # Primero se paga 0,10 % por entrar y luego el activo sube 10 %.
    assert reward == pytest.approx((1 - 0.001) * 1.10 - 1, rel=1e-12)


def test_cambiar_de_posicion_sin_ganancia_destruye_capital():
    # Con el precio constante, cualquier cambio de posición solo genera costos.
    env = make_env([100.0] * 11, cost_bps=10.0)
    infos = run_episode(env, [2, 0, 2, 0, 2, 0, 2, 0, 2, 0])
    assert infos[-1]["portfolio_value"] < CAPITAL
    assert infos[-1]["total_cost"] == pytest.approx(
        CAPITAL - infos[-1]["portfolio_value"], rel=1e-9
    )


# ----------------------------------------------------------------------
# Mecánica del episodio
# ----------------------------------------------------------------------
def test_episodio_completo_termina_con_truncated_en_el_ultimo_paso():
    prices = random_walk(50)
    env = make_env(prices)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    pasos = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(int(rng.integers(0, 3)))
        pasos += 1
        assert np.isfinite(reward)
        assert obs.shape == env.observation_space.shape
    assert pasos == len(prices) - 1
    assert truncated and not terminated


def test_step_despues_del_final_exige_reset():
    env = make_env([100.0, 101.0])
    env.reset(seed=0)
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_accion_invalida_lanza_error():
    env = make_env([100.0, 101.0, 102.0])
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(3)


def test_la_posicion_aparece_en_la_observacion():
    env = make_env(random_walk(10))
    obs, _ = env.reset(seed=0)
    assert obs[-1] == 0.0
    obs, *_ = env.step(1)
    assert obs[-1] == 0.5
    obs, *_ = env.step(2)
    assert obs[-1] == 1.0


def test_reset_deja_el_entorno_como_nuevo():
    env = make_env(random_walk(30))
    run_episode(env, [2] * 10)
    obs, info = env.reset(seed=0)
    assert info["portfolio_value"] == CAPITAL
    _, _, _, _, info = env.step(2)
    assert info["n_trades"] == 1


def test_arranque_aleatorio_es_reproducible_con_la_misma_semilla():
    prices = random_walk(300)
    env_a = make_env(prices, episode_length=50, random_start=True)
    env_b = make_env(prices, episode_length=50, random_start=True)
    _, info_a = env_a.reset(seed=7)
    _, info_b = env_b.reset(seed=7)
    assert info_a["index"] == info_b["index"]
    pasos = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env_a.step(1)
        done = term or trunc
        pasos += 1
    assert pasos == 50


def test_historial_registra_el_recorrido():
    prices = random_walk(20)
    env = make_env(prices, record_history=True)
    run_episode(env, [2] * 19)
    history = env.get_history()
    assert len(history["portfolio_value"]) == 19
    assert history["position"][0] == 1.0


# ----------------------------------------------------------------------
# Recompensa intercambiable
# ----------------------------------------------------------------------
def test_recompensa_personalizada_recibe_info_y_se_reinicia():
    class Contador:
        def __init__(self):
            self.resets = 0
            self.llamadas = 0

        def reset(self):
            self.resets += 1

        def __call__(self, info):
            self.llamadas += 1
            assert "net_return" in info and "drawdown" in info
            return -1.0

    recompensa = Contador()
    env = make_env(random_walk(10), reward_fn=recompensa)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(1)
    assert reward == -1.0
    assert recompensa.resets == 1 and recompensa.llamadas == 1


def test_recompensa_por_defecto_es_retorno_neto():
    env = make_env(random_walk(10))
    assert isinstance(env.reward_fn, NetReturnReward)


# ----------------------------------------------------------------------
# Lookahead
# ----------------------------------------------------------------------
def test_modificar_el_futuro_no_cambia_lo_ya_ocurrido():
    prices = random_walk(100)
    returns = np.concatenate([[0.0], prices[1:] / prices[:-1] - 1.0])
    features = np.column_stack([returns, returns**2])
    k = 40  # se ejecutan k pasos y después se comparan los resultados

    prices_alt = prices.copy()
    features_alt = features.copy()
    # Se altera todo lo que ocurre después del precio k, de tal manera que un
    # entorno con lookahead produciría resultados distintos.
    prices_alt[k + 1 :] *= 7.0
    features_alt[k + 1 :] = 999.0

    acciones = np.random.default_rng(3).integers(0, 3, size=k)
    env_a = make_env(prices, features=features)
    env_b = make_env(prices_alt, features=features_alt)
    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)
    assert np.array_equal(obs_a, obs_b)

    for accion in acciones:
        obs_a, rew_a, _, _, info_a = env_a.step(int(accion))
        obs_b, rew_b, _, _, info_b = env_b.step(int(accion))
        assert np.array_equal(obs_a, obs_b)
        assert rew_a == rew_b
        assert info_a["portfolio_value"] == info_b["portfolio_value"]


def test_la_observacion_en_t_no_depende_del_precio_siguiente():
    prices = random_walk(20)
    prices_alt = prices.copy()
    prices_alt[1] = prices[1] * 5.0
    returns = np.zeros(20)
    env_a = make_env(prices, features=np.column_stack([returns, returns]))
    env_b = make_env(prices_alt, features=np.column_stack([returns, returns]))
    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)
    assert np.array_equal(obs_a, obs_b)


# ----------------------------------------------------------------------
# Validación de entradas
# ----------------------------------------------------------------------
def test_features_con_nan_se_rechazan():
    prices = random_walk(10)
    features = np.zeros((10, 2))
    features[3, 0] = np.nan
    with pytest.raises(ValueError):
        make_env(prices, features=features)


def test_no_se_permite_apalancamiento_ni_cortos():
    with pytest.raises(ValueError):
        make_env(random_walk(10), action_fractions=(-1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        make_env(random_walk(10), action_fractions=(0.0, 1.0, 2.0))


def test_costo_negativo_se_rechaza():
    with pytest.raises(ValueError):
        make_env(random_walk(10), cost_bps=-1.0)


# ----------------------------------------------------------------------
# Compatibilidad con Gymnasium y Stable-Baselines3
# ----------------------------------------------------------------------
def test_check_env_de_stable_baselines3():
    env_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    env = make_env(random_walk(200))
    env_checker.check_env(env, warn=True)


# ----------------------------------------------------------------------
# Integración con los datos reales del proyecto
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def datos_reales():
    from pathlib import Path

    if not (Path(__file__).resolve().parents[1] / "data" / "binance-BTCUSDT-1h-interpolated.pkl").exists():
        pytest.skip("Falta el dataset interpolado, por lo que se debe ejecutar python -m src.prepare_data.")
    from src.env_factory import build_market_data

    datos, _ = build_market_data()
    return datos


def test_datos_reales_sin_nan_y_con_particiones_ordenadas(datos_reales):
    for parte in datos_reales.values():
        assert np.all(np.isfinite(parte.features))
        assert len(parte.prices) == len(parte.features) == len(parte.timestamps)
    assert datos_reales["train"].timestamps[-1] < datos_reales["validation"].timestamps[0]
    assert datos_reales["validation"].timestamps[-1] < datos_reales["held_out"].timestamps[0]


def test_escalador_ajustado_solo_con_entrenamiento(datos_reales):
    train = datos_reales["train"].features
    # En entrenamiento las variables quedan con media 0 y desviación estándar 1.
    assert np.allclose(train.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(train.std(axis=0), 1.0, atol=1e-8)
    # En validación esto no ocurre, dado que el escalador no se vuelve a ajustar.
    validacion = datos_reales["validation"].features
    assert not np.allclose(validacion.mean(axis=0), 0.0, atol=1e-3)


def test_comprar_y_mantener_en_validacion_real(datos_reales):
    parte = datos_reales["validation"]
    env = TradingEnvironment(
        prices=parte.prices,
        features=parte.features,
        initial_capital=CAPITAL,
        transaction_cost_bps=10.0,
    )
    infos = run_episode(env, [2] * (len(parte.prices) - 1))
    esperado = CAPITAL * (1 - 0.001) * parte.prices[-1] / parte.prices[0]
    assert infos[-1]["portfolio_value"] == pytest.approx(esperado, rel=1e-9)
