import jax.numpy as jnp

MSUN_SECONDS = 4.925490947e-6
MPC_METERS = 3.0856775814913673e22
C = 299792458.0

def chirp_mass(m1, m2):
    m1 = jnp.asarray(m1)
    m2 = jnp.asarray(m2)
    mtot = m1 + m2
    eta = m1 * m2 / mtot**2
    return mtot * eta ** (3.0 / 5.0)


def eta_from_masses(m1, m2):
    mtot = m1 + m2
    return m1 * m2 / mtot**2


def taylorf2_0pn_phase(f, m1, m2, tc=0.0, phic=0.0):
    f = jnp.asarray(f)

    mtot = m1 + m2
    eta = eta_from_masses(m1, m2)
    mtot_sec = mtot * MSUN_SECONDS

    v = (jnp.pi * mtot_sec * f) ** (1.0 / 3.0)

    psi = (
        2.0 * jnp.pi * f * tc
        - phic
        - jnp.pi / 4.0
        + 3.0 / (128.0 * eta) * v ** (-5.0)
    )

    return psi


def taylorf2_0pn(
    f,
    m1,
    m2,
    distance_mpc=1.0,
    tc=0.0,
    phic=0.0,
    amplitude=1.0,
    f_ref=100.0,
):
    f = jnp.asarray(f)

    psi = taylorf2_0pn_phase(f, m1, m2, tc=tc, phic=phic)

    mc_sec = chirp_mass(m1, m2) * MSUN_SECONDS
    distance_sec = distance_mpc * MPC_METERS / C

    amp = (
        jnp.sqrt(5.0 / 24.0)
        * jnp.pi ** (-2.0 / 3.0)
        * mc_sec ** (5.0 / 6.0)
        / distance_sec
        * f ** (-7.0 / 6.0)
    )

    return amp * jnp.exp(1j * psi)



def get_taylor_signal(
    f,
    log10_distance_mpc,
    log10_mass,
):
    m1 = 10 ** log10_mass
    distance_mpc = 10 ** log10_distance_mpc

    m2 = m1

    return taylorf2_0pn(
        f=f,
        m1=m1,
        m2=m2,
        distance_mpc=distance_mpc,
    )


def lisa_asd(f):
    f = jnp.asarray(f)

    L = 2.5e9
    fstar = C / (2.0 * jnp.pi * L)

    p_oms = 1.5e-11**2 * (1.0 + (2.0e-3 / f) ** 4)

    p_acc = (
        3.0e-15**2
        * (1.0 + (0.4e-3 / f) ** 2)
        * (1.0 + (f / 8.0e-3) ** 4)
        / (2.0 * jnp.pi * f) ** 4
    )

    psd = (
        10.0 / (3.0 * L**2)
        * (p_oms + 4.0 * p_acc)
        * (1.0 + 0.6 * (f / fstar) ** 2)
    )

    return jnp.sqrt(psd)
