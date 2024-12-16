"""
Defines PDE problems to solve

Each problem class must inherit from the Problem base class.
Each problem class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax.nn
import jax.numpy as jnp
import numpy as np

from fbpinns.util.logger import logger
from fbpinns.traditional_solutions.analytical.burgers_solution import burgers_viscous_time_exact1
from fbpinns.traditional_solutions.seismic_cpml.seismic_CPML_2D_pressure_second_order import seismicCPML2D


class Problem:
    """Base problem class to be inherited by different problem classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""

        # below parameters need to be defined
        static_params = {
            "dims":None,# (ud, xd)# dimensionality of u and x
            }
        raise NotImplementedError

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        """Samples all constraints.
        Returns [[x_batch, *any_constraining_values, required_ujs], ...]. Each list element contains
        the x_batch points and any constraining values passed to the loss function, and the required
        solution and gradient components required in the loss function, for each constraint."""
        raise NotImplementedError

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """Applies optional constraining operator"""
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        """Computes the PINN loss function, using constraints with the same structure output by sample_constraints"""
        raise NotImplementedError

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """Defines exact solution, if it exists"""
        raise NotImplementedError


class HarmonicOscillator1D(Problem):
    """Solves the time-dependent damped harmonic oscillator
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu":mu,
            "k":k,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )

        # boundary loss
        x_batch_boundary = jnp.array([0.]).reshape((1,1))
        u_boundary = jnp.array([1.]).reshape((1,1))
        ut_boundary = jnp.array([0.]).reshape((1,1))
        required_ujs_boundary = (
            (0,()),
            (0,(0,)),
        )

        return [[x_batch_phys, required_ujs_phys], [x_batch_boundary, u_boundary, ut_boundary, required_ujs_boundary]]

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics loss
        _, u, ut, utt = constraints[0]
        phys = jnp.mean((utt + mu*ut + k*u)**2)

        # boundary loss
        _, uc, utc, u, ut = constraints[1]
        if len(uc):
            boundary = 1e6*jnp.mean((u-uc)**2) + 1e2*jnp.mean((ut-utc)**2)
        else:
            boundary = 0# if no boundary points are inside the active subdomains (i.e. u.shape[0]=0), jnp.mean returns nan

        return phys + boundary

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):

        d, w0 = all_params["static"]["problem"]["d"], all_params["static"]["problem"]["w0"]

        w = jnp.sqrt(w0**2-d**2)
        phi = jnp.arctan(-d/w)
        A = 1/(2*jnp.cos(phi))
        cos = jnp.cos(phi + w * x_batch)
        exp = jnp.exp(-d * x_batch)
        u = exp * 2 * A * cos

        return u


class HarmonicOscillator1DHardBC(HarmonicOscillator1D):
    """Solves the time-dependent damped harmonic oscillator using hard boundary conditions
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20, sd=0.1):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu":mu,
            "k":k,
            "sd":sd,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )
        return [[x_batch_phys, required_ujs_phys],]# only physics loss required in this case

    @staticmethod
    def constraining_fn(all_params, x_batch, u):

        sd = all_params["static"]["problem"]["sd"]
        x, tanh = x_batch[:,0:1], jnp.tanh

        u = 1 + (tanh(x/sd)**2) * u# applies hard BCs
        return u

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["static"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics loss
        _, u, ut, utt = constraints[0]
        phys = jnp.mean((utt + mu*ut + k*u)**2)

        return phys


class HarmonicOscillator1DInverse(HarmonicOscillator1D):
    """Solves the time-dependent damped harmonic oscillator inverse problem
          d^2 u      du
        m ----- + mu -- + ku = 0
          dt^2       dt

        Boundary conditions:
        u (0) = 1
        u'(0) = 0
    """

    @staticmethod
    def init_params(d=2, w0=20):

        mu, k = 2*d, w0**2

        static_params = {
            "dims":(1,1),
            "d":d,
            "w0":w0,
            "mu_true":mu,
            "k":k,
            }
        trainable_params = {
            "mu":jnp.array(0.),# learn mu from constraints
            }

        return static_params, trainable_params

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0))
        )

        # data loss
        x_batch_data = jnp.linspace(0,1,13).astype(float).reshape((13,1))# use 13 observational data points
        u_data = HarmonicOscillator1DInverse.exact_solution(all_params, x_batch_data)
        required_ujs_data = (
            (0,()),
            )

        return [[x_batch_phys, required_ujs_phys], [x_batch_data, u_data, required_ujs_data]]

    @staticmethod
    def loss_fn(all_params, constraints):

        mu, k = all_params["trainable"]["problem"]["mu"], all_params["static"]["problem"]["k"]

        # physics loss
        _, u, ut, utt = constraints[0]
        phys = jnp.mean((utt + mu*ut + k*u)**2)

        # data loss
        _, uc, u = constraints[1]
        data = 1e6*jnp.mean((u-uc)**2)

        return phys + data


class BurgersEquation2D(Problem):
    """Solves the time-dependent 1D viscous Burgers equation
        du       du        d^2 u
        -- + u * -- = nu * -----
        dt       dx        dx^2

        for -1.0 < x < +1.0, and 0 < t

        Boundary conditions:
        u(x,0) = - sin(pi*x)
        u(-1,t) = u(+1,t) = 0
    """

    @staticmethod
    def init_params(nu=0.01/jnp.pi, sd=0.1):

        static_params = {
            "dims":(1,2),
            "nu":nu,
            "sd":sd,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(1,)),
            (0,(0,0)),
        )
        return [[x_batch_phys, required_ujs_phys],]


    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        x, t, tanh, sin, pi, exp = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh, jnp.sin, jnp.pi, jnp.exp
        u = tanh((x+1)/sd)*tanh((1-x)/sd)*tanh((t-0)/sd)*u - sin(pi*x)
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        nu = all_params["static"]["problem"]["nu"]
        _, u, ux, ut, uxx = constraints[0]
        phys = ut + (u*ux) - (nu*uxx)
        return jnp.mean(phys**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        nu = all_params["static"]["problem"]["nu"]

        # use the burgers_solution code to compute analytical solution
        xmin,xmax = x_batch[:,0].min().item(), x_batch[:,0].max().item()
        tmin,tmax = x_batch[:,1].min().item(), x_batch[:,1].max().item()
        vx = np.linspace(xmin,xmax,batch_shape[0])
        vt = np.linspace(tmin,tmax,batch_shape[1])
        logger.info("Running burgers_viscous_time_exact1..")
        vu = burgers_viscous_time_exact1(nu, len(vx), vx, len(vt), vt)
        u = jnp.array(vu.flatten()).reshape((-1,1))
        return u


class Heart(Problem):
    """
    """

    @staticmethod
    def init_params(boundary, E=1):

        static_params = {
            "dims":(1,2),
            "E":E,
            "boundary":boundary,
        }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),   #u1
            (0,(0,0)),#u1_xx
            (0,(1,1)),#u1_yy
            # (0,(0,)),  #u1_x
            # (0,(1,)),  #u1_y
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        boundary = all_params["static"]["problem"]["boundary"]
                
        u = u * boundary
        # u = u / jnp.mean(u**2)
        
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        E = all_params["static"]["problem"]["E"]
        x_batch, u1, u1_xx, u1_yy = constraints[0]
        x, y, tanh = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh
        # c = (50 / 100)
        # a = (45 / 100)
        # b = (42 / 100)
        # boundary = (((x - c)/(b))**2 * ((y - c)/(b))**3 - (((x - c)/b)**2 + ((y - a)/(b))**2 - 1)**3)
        # cond_outside = (boundary < 0)
        
        # smooth_loss_x = u1_x * ~cond_outside
        # smooth_loss_y = u1_y * ~cond_outside
        
        phys1 = u1_xx + u1_yy + (E * u1)

        return jnp.mean(phys1**2)# + jnp.mean(smooth_loss_x**2) + jnp.mean(smooth_loss_y**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        # s = x_batch[:, 0]
        return x_batch


class Schrodinger2D(Problem):
    """
    """

    @staticmethod
    def init_params(sd=0.1, E=1):

        static_params = {
            "dims":(1,2),
            "sd":sd,
            "E":E,
            # "hbar/2me":5.788382e-5
        }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),   #u1
            (0,(0,0)),#u1_xx
            (0,(1,1)),#u1_yy
            # (1,()),   #u2
            # (1,(0,0)),#u2_xx
            # (1,(1,1)),#u2_yy
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        
        x, y, tanh = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh
        
        u1 = u
        # u1 = u[:, 0:1]
        # u2 = u[:, 1:2]
        
        boundary = tanh((x)/sd)*tanh((1-x)/sd)*tanh((y)/sd)*tanh((1-y)/sd)
        
        u1 = boundary * u1
        # u2 = boundary * u2
        
        u1 = u1 / jnp.sqrt(jnp.mean(u1**2))
        # u2 = u2 / jnp.sqrt(jnp.mean(u2**2))
        
        return u1
        # return jnp.concatenate((u1, u2), axis=1)

    @staticmethod
    def loss_fn(all_params, constraints):
        E = all_params["static"]["problem"]["E"]
        # x_batch, u1, u1_xx, u1_yy, u2, u2_xx, u2_yy = constraints[0]
        x_batch, u1, u1_xx, u1_yy = constraints[0]
        
        phys1 = u1_xx + u1_yy + E * u1
        # phys2 = u2_xx + u2_yy + E * u2

        # orthogonality = jnp.abs(np.sum(u1 * u2))
        
        # penalty = jnp.abs(np.sum(u1 * u2)**2) / 100

        return jnp.mean(phys1**2)# + jnp.mean(phys2**2) + orthogonality# + penalty

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        # s = x_batch[:, 0]
        return x_batch


class Laplace2D(Problem):
    """
        d^2 u   d^2 u   
        ----- + ----- = -f
        dx^2    dy^2    

        Boundary conditions:
        u(x,0) = 0
        u(x,3) = 0
        u(0,y) = 0
        u(3,y) = 0
    """

    @staticmethod
    def init_params(sd=0.1, n=3):

        static_params = {
            "dims":(1,2),
            "sd":sd,
            "n":n,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        
        x, y, tanh = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh
        u = tanh((x)/sd)*tanh((3-x)/sd)*tanh((y)/sd)*tanh((3-y)/sd)*u
        
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        n = all_params["static"]["problem"]["n"]
        x_batch, u_xx, u_yy = constraints[0]
        x, y, pi, sin = x_batch[:,0], x_batch[:,1], jnp.pi, jnp.sin
        
        f = jnp.zeros(x_batch.shape[0])
        for w in (2**i * pi for i in range(n)):
            f += w**2 * sin(w * x) * sin(w * y)
        f *= 2 / n
        
        phys = f.reshape((-1, 1)) + u_xx + u_yy

        return jnp.mean(phys**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        n = all_params["static"]["problem"]["n"]
        x, y, pi, sin = x_batch[:,0], x_batch[:,1], jnp.pi, jnp.sin
        
        s = jnp.zeros(x_batch.shape[0])
        for w in (2**i * pi for i in range(n)):
            s += sin(w * x) * sin(w * y)

        return (1 / n * s).reshape((-1,1))


from jax import custom_vjp


class GravityHeart(Problem):
    """
        d^2 u   d^2 u   
        ----- + ----- = rho(x)
        dx^2    dy^2    

        Boundary conditions:
        u(x,0) = 0
        u(x,1) = 0
        u(0,y) = 0
        u(1,y) = 0
    """

    @staticmethod
    def init_params(ansatz, ansatz_x, ansatz_y, ansatz_xx, ansatz_yy):

        static_params = {
            "dims":(1,2),
            "ansatz":ansatz,
            "ansatz_x":ansatz_x,
            "ansatz_y":ansatz_y,
            "ansatz_xx":ansatz_xx,
            "ansatz_yy":ansatz_yy,
            # "ansatz_xx":ansatz,
            "sources": [
                # [np.array([0.0, 0.0]), 0.1, 8],
                [np.array([0.3, 0.7]), 0.1, 8],
                [np.array([-0.7, 0.3]), 0.1, 2],
                [np.array([-0.3, -0.7]), 0.1, 1],
                [np.array([0.7, -0.3]), 0.1, 4]
            ]
        }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        ansatz = all_params["static"]["problem"]["ansatz"]
        ansatz_x = all_params["static"]["problem"]["ansatz_x"]
        ansatz_y = all_params["static"]["problem"]["ansatz_y"]
        ansatz_xx = all_params["static"]["problem"]["ansatz_xx"]
        ansatz_yy = all_params["static"]["problem"]["ansatz_yy"]        
        
        @jax.custom_jvp
        def ansatz_x_fn(x, y):
            return ansatz_x

        @ansatz_x_fn.defjvp
        def ansatz_x_fn_jvp(primals, tangents):
            xs, ys = primals
            x_dot, y_dot = tangents
            primal_out = ansatz_x
            tangent_out = ansatz_xx * x_dot
            return primal_out, tangent_out
        
        @jax.custom_jvp
        def ansatz_y_fn(x, y):
            return ansatz_y

        @ansatz_y_fn.defjvp
        def ansatz_y_fn_jvp(primals, tangents):
            xs, ys = primals
            x_dot, y_dot = tangents
            primal_out = ansatz_y
            tangent_out = ansatz_yy * y_dot
            return primal_out, tangent_out        
        
        @jax.custom_jvp
        def ansatz_fn(x, y):
            return ansatz

        @ansatz_fn.defjvp
        def ansatz_fn_jvp(primals, tangents):
            xs, ys = primals
            x_dot, y_dot = tangents
            
            # Call the derivative functions to compute the tangent
            ansatz_x_value = ansatz_x_fn(xs, ys)  # Call ansatz_x_fn
            ansatz_y_value = ansatz_y_fn(xs, ys)  # Call ansatz_y_fn
            
            primal_out = ansatz
            tangent_out = ansatz_x_value * x_dot + ansatz_y_value * y_dot
            return primal_out, tangent_out

        x, y = x_batch[:,0:1], x_batch[:,1:2]
        
        bc = ansatz_fn(x, y)
        # bc = ansatz

        return jnp.multiply(bc, u)

    @staticmethod
    def loss_fn(all_params, constraints):
        sources = all_params["static"]["problem"]["sources"]
        
        x_batch, u_xx, u_yy = constraints[0]
                
        norm = jnp.linalg.norm
        
        f = jnp.zeros(x_batch.shape[0])
        
        for source in sources:
            f += (norm(x_batch - source[0], axis=1) < source[1]) * source[2]

        phys = f.reshape((-1, 1)) + u_xx + u_yy

        return jnp.mean(phys**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        s = x_batch[:, 0]
        return s


class Gravity(Problem):
    """
        d^2 u   d^2 u   
        ----- + ----- = rho(x)
        dx^2    dy^2    

        Boundary conditions:
        u(x,0) = 0
        u(x,1) = 0
        u(0,y) = 0
        u(1,y) = 0
    """

    @staticmethod
    def init_params(sd=0.1):

        static_params = {
            "dims":(1,2),
            "sd":sd,
            # "sources": [
            #     [np.array([0.3, 0.7]), 0.1, 8],
            #     [np.array([-0.7, 0.3]), 0.1, 2],
            #     [np.array([-0.3, -0.7]), 0.1, 1],
            #     [np.array([0.7, -0.3]), 0.1, 4]
            # ]
            "sources": [
                [np.array([0.3, 0.7]), 0.1, 1000],
                # [np.array([-0.7, 0.3]), 0.1, 2],
                [np.array([-0.3, -0.7]), 0.1, 1],
                # [np.array([0.7, -0.3]), 0.1, 4]
            ]
        }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        
        # SQUARE
        # x, y, tanh = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh
        # u = tanh((x+1)/sd)*tanh((1-x)/sd)*tanh((y+1)/sd)*tanh((1-y)/sd)*u
        
        # DONUT
        x, y, tanh, sqrt = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh, jnp.sqrt
        r = sqrt(x**2 + y**2)
        u = tanh((r - 0.5)/sd)*tanh((1-r)/sd) * u
        
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        sources = all_params["static"]["problem"]["sources"]
        x_batch, u_xx, u_yy = constraints[0]

        norm = jnp.linalg.norm
        
        f = jnp.zeros(x_batch.shape[0])
        
        for source in sources:
            f += (norm(x_batch - source[0], axis=1) < source[1]) * source[2]

        phys = f.reshape((-1, 1)) + u_xx + u_yy

        return jnp.mean(phys**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        s = x_batch[:, 0]
        return s

class GravityDonut(Problem):
    """
        d^2 u   d^2 u   
        ----- + ----- = rho(x)
        dx^2    dy^2    

        Boundary conditions:
        u(x,0) = 0
        u(x,1) = 0
        u(0,y) = 0
        u(1,y) = 0
    """

    @staticmethod
    def init_params(sd=0.1):

        static_params = {
            "dims":(1,2),
            "sd":sd,
            "sources": [
                [np.array([0.3, 0.7]), 0.1, 10],
                [np.array([-0.7, 0.3]), 0.1, 3],
                [np.array([-0.3, -0.7]), 0.1, 1],
                [np.array([0.7, -0.3]), 0.1, 3]
            ]
        }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        
        x, y, tanh, sqrt, square = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh, jnp.sqrt, jnp.square
        r = sqrt(square(x) + square(y))
        u = tanh((r - 0.5) / sd) * tanh((1- r) / sd) * u
        cond_l = r > 0.5
        cond_r = r < 1.0
        u = u * (cond_l) * (cond_r)
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        sources = all_params["static"]["problem"]["sources"]
        x_batch, u_xx, u_yy = constraints[0]
        x, y, = x_batch[:,0], x_batch[:,1]

        norm = jnp.linalg.norm
        
        f = jnp.zeros(x_batch.shape[0])
        
        for source in sources:
            f += (norm(x_batch - source[0], axis=1) < source[1]) * source[2]

        phys = f.reshape((-1, 1)) + u_xx + u_yy

        return jnp.mean(phys**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        s = x_batch[:, 0]
        return s

class Laplace2D_2(Problem):
    """
        d^2 u   d^2 u   
        ----- + ----- = -f
        dx^2    dy^2    

        Boundary conditions:
        u(x,0) = sinh(x)
        u(x,pi) = -sinh(x)
        u(0,y) = 0
        u(pi,y) = sinh(pi)cos(y)
    """

    @staticmethod
    def init_params(sd=0.1):

        static_params = {
            "dims":(1,2),
            "sd":sd,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        
        x, y, tanh, sinh, cos, pi = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh, jnp.sinh, jnp.cos, jnp.pi
        tanh_yl = tanh(y/sd)
        tanh_yr = tanh((pi-y)/sd)
        tanh_xl = tanh(x/sd)
        tanh_xr = tanh((pi-x)/sd)
        #   ---u doesnt contribute at the boundary--- + -------bottom---------- + --------top------------- + ----------right------------------
        u = tanh_yl * tanh_yr * tanh_xl * tanh_xr * u + sinh(x) * (1 - tanh_yl) + -sinh(x) * (1 - tanh_yr) + sinh(pi) * cos(y) * (1 - tanh_xr)

        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        x_batch, u_xx, u_yy = constraints[0]

        phys = u_xx + u_yy

        return jnp.mean(phys**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        x, y, pi, sinh, cos = x_batch[:,0], x_batch[:,1], jnp.pi, jnp.sinh, jnp.cos
        
        s = sinh(x) * cos(y)

        return s.reshape((-1,1))

class Ice(Problem):
    """
        d^2 u   d^2 u   
        ----- + ----- = -f
        dx^2    dy^2    

        Boundary conditions:
        u(x,0) = sinh(x)
        u(x,pi) = -sinh(x)
        u(0,y) = 0
        u(pi,y) = sinh(pi)cos(y)
    """

    @staticmethod
    def init_params(sdx=0.01, sdt=1, c_ice=2050, c_water=4184, rho_ice=910, rho_water=1000,
                    k_ice=2.2, k_water=0.6, L_water=334, freeze_temp=0, zero_at_h_weight=1., h_x_weight=1.):

        static_params = {
            "dims":(2,2),
            "sdx":sdx,
            "sdt":sdt,
            "c_ice":c_ice,
            "c_water":c_water,
            "rho_ice":rho_ice,
            "rho_water":rho_water,
            "k_ice":k_ice,
            "k_water":k_water,
            "L_water":L_water,
            "freeze_temp":freeze_temp,
            "zero_at_h_weight":zero_at_h_weight,
            "h_x_weight":h_x_weight,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),
            (0,(0,)),
            (0,(0,0)),
            (0,(1,)),
            (1,()),
            (1,(0,)),
            (1,(1,)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sdx = all_params["static"]["problem"]["sdx"]
        sdt = all_params["static"]["problem"]["sdt"]
        
        x, t, T, h, tanh = x_batch[:,0:1], x_batch[:,1:2], u[:,0:1], u[:,1:2], jax.nn.tanh
        tanh_x0 = tanh(x/sdx)
        tanh_xR = tanh((0.02 - x)/sdx)
        tanh_t0 = tanh(t/sdt)

        T = tanh_x0 * tanh_xR * tanh_t0 * T + 10 * (1 - (tanh_xR * tanh_t0)) + -10 * (1 - tanh_x0)
        
        h = tanh_t0 * h
        
        return jnp.concatenate((T, h), axis=1)

    @staticmethod
    def loss_fn(all_params, constraints):
        c_ice = all_params["static"]["problem"]["c_ice"]
        c_water = all_params["static"]["problem"]["c_water"]
        rho_ice = all_params["static"]["problem"]["rho_ice"]
        rho_water = all_params["static"]["problem"]["rho_water"]
        k_ice = all_params["static"]["problem"]["k_ice"]
        k_water = all_params["static"]["problem"]["k_water"]
        L_water = all_params["static"]["problem"]["L_water"]
        freeze_temp = all_params["static"]["problem"]["freeze_temp"]
        zero_at_h_weight = all_params["static"]["problem"]["zero_at_h_weight"]
        h_x_weight = all_params["static"]["problem"]["h_x_weight"]
        
        # factor = rho_water * L_water / k_water
        
        x_batch, T, T_x, T_xx, T_t, h, h_x, h_t = constraints[0]
        x, t = x_batch[:,0], x_batch[:,1]
        
        cond_at_h = (x > h - 1e-2) & (x < h + 1e-2)
        cond_ice = (x < h)
        cond_water = (x >= h)
        
        loss_heateqn_ice = (rho_ice * c_ice * T_t * (~cond_ice) - k_ice * T_xx * (~cond_ice)) / 10_000.
        
        loss_heateqn_water = (rho_water * c_water * T_t * (~cond_water) - k_ice * T_xx * (~cond_water)) / 10_000.
        
        loss_stefan = (rho_water * L_water * h_t * (~cond_at_h) + k_water * T_x * (~cond_at_h)) / 100_000.

        loss_T_at_h_is_zero = T * (~cond_at_h) - freeze_temp

        loss_h_not_in_domain = jnp.clip(-h, 0, None)
        
        loss_h_not_in_domain2 = jnp.clip(0.01 * h, 0, None)

        return (jnp.mean(loss_heateqn_ice**2) + jnp.mean(loss_heateqn_water**2) + jnp.mean(loss_stefan**2)
                + h_x_weight * jnp.mean(h_x**2) + zero_at_h_weight * jnp.mean(loss_T_at_h_is_zero**2) + h_x_weight * jnp.mean(loss_h_not_in_domain**2)  + h_x_weight * jnp.mean(loss_h_not_in_domain2**2)
        )

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        x, t = x_batch[:,0], x_batch[:,1]
        
        s = 0.001 * x

        return s.reshape((-1,1))


class CahnHilliard(Problem):
    """

    """

    @staticmethod
    def init_params(sdt=0.1, gamma=0.5):
        static_params = {
            "dims":(1,3),
            "sdt":sdt,
            "gamma":gamma,
            }

        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,()),  # u
            (0,(2,)),  # u_t
            (0,(0,)),  # u_x
            (0,(0,0)),  # u_xx
            (0,(0,0,0)),  # u_xxx
            (0,(0,0,0,0)),  # u_xxxx
            (0,(1,)),  # u_y
            (0,(1,1)),  # u_yy
            (0,(1,1,1)),  # u_yy
            (0,(1,1,1,1)),  # u_yyyy
            (0,(0,0,1)),  # u_xxy
            (0,(0,1,1)),  # u_xyy
            (0,(0,0,1,1)),  # u_xxyy
        )
        
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sdt = all_params["static"]["problem"]["sdt"]
        x, y, t, tanh, sin, cos = x_batch[:,0:1], x_batch[:,1:2], x_batch[:,2:3], jax.nn.tanh, jnp.sin, jnp.cos
        
        noise = 0.25 * jax.random.normal(jax.random.PRNGKey(42), shape=u.shape)
        
        tanh_t0 = tanh(t / sdt)
        
        print("shapes", u.shape, noise.shape)
        
        return tanh_t0 * u + (1 - tanh_t0) * noise

    @staticmethod
    def loss_fn(all_params, constraints):
        gamma = all_params["static"]["problem"]["gamma"]
        
        x_batch, u, u_t, u_x, u_xx, u_xxx, u_xxxx, u_y, u_yy, u_yyy, u_yyyy, u_xxy, u_xyy, u_xxyy = constraints[0]
        
        x, y, t = x_batch[:,0:1], x_batch[:,1:2], x_batch[:,2:3]
        
        # cond_x0 = x == 0
        # cond_x1 = x == 100
        # cond_y0 = y == 0
        # cond_y1 = y == 100
        
        u2 = u * u
        
        cahnhilliard_loss = u_t + gamma * (u_xxxx + 2 * u_xxyy + u_yyyy) - 6 * (u_x**2 + u_y**2) - 3 * u2 * (u_xx + u_yy)

        # mu_x = gamma * (u_xxx + u_xyy) + u_x * (1 - 3 * u2)
        # mu_y = gamma * (u_xxy + u_yyy) + u_x * (1 - 3 * u2)
        
        # u_x0 = u_x * (~cond_x0)
        # u_x1 = u_x * (~cond_x1)
        # u_y0 = u_y * (~cond_y0)
        # u_y1 = u_y * (~cond_y1)
        
        # mu_x0 = mu_x * (~cond_x0)
        # mu_x1 = mu_x * (~cond_x1)
        # mu_y0 = mu_y * (~cond_y0)
        # mu_y1 = mu_y * (~cond_y1)
        
        return (
            jnp.mean(cahnhilliard_loss**2) #+
            # jnp.mean(mu_x0**2) +
            # jnp.mean(mu_x1**2) +
            # jnp.mean(mu_y0**2) +
            # jnp.mean(mu_y1**2) +
            # jnp.mean(u_x0**2) +
            # jnp.mean(u_x1**2) +
            # jnp.mean(u_y0**2) +
            # jnp.mean(u_y1**2)
        )

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        x, t = x_batch[:,0], x_batch[:,1]

        return x.reshape((-1,1))


class WaveEquationConstantVelocity3D(Problem):
    """Solves the time-dependent (2+1)D wave equation with constant velocity
        d^2 u   d^2 u    1  d^2 u
        ----- + ----- - --- ----- = f
        dx^2    dy^2    c^2 dt^2
    """

    @staticmethod
    def init_params(sd=1, source=np.array([[0., 0., 0.2, 1.]])):

        static_params = {
            "dims":(1,3),
            "sd":0.05,
            }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):

        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0,(0,0)),
            (0,(1,1)),
            (0,(2,2)),
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        sd = all_params["static"]["problem"]["sd"]
        
        x, y, tanh = x_batch[:,0:1], x_batch[:,1:2], jax.nn.tanh
        u = tanh((x+1)/sd)*tanh((1-x)/sd)*tanh((y+1)/sd)*tanh((1-y)/sd)*u
        
        return u
        
        # c0, source = params["c0"], params["source"]
        # x, t = x_batch[:,0:2], x_batch[:,2:3]
        # tanh, exp = jax.nn.tanh, jnp.exp

        # # get starting wavefield
        # p = jnp.expand_dims(source, axis=1)# (k, 1, 4)
        # x = jnp.expand_dims(x, axis=0)# (1, n, 2)
        # f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)

        # # form time-decaying anzatz
        # t1 = source[:,2].min()/c0
        # f = exp(-0.5*(1.5*t/t1)**2) * f
        t = tanh(2.5*t/t1)**2
        return f + t*u

    @staticmethod
    def loss_fn(all_params, constraints):
        c_fn = all_params["static"]["problem"]["c_fn"]
        x_batch, uxx, uyy, utt = constraints[0]
        phys = (uxx + uyy) - (1/c_fn(all_params, x_batch)**2)*utt
        return jnp.mean(phys**2)

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):

        # use the seismicCPML2D FD code with very fine sampling to compute solution

        params = all_params["static"]["problem"]
        c0, source = params["c0"], params["source"]
        c_fn = params["c_fn"]

        (xmin, ymin, tmin), (xmax, ymax, tmax) = np.array(x_batch.min(0)), np.array(x_batch.max(0))

        # get grid spacing
        deltax, deltay, deltat = (xmax-xmin)/(batch_shape[0]-1), (ymax-ymin)/(batch_shape[1]-1), (tmax-tmin)/(batch_shape[2]-1)

        # get f0, target deltas of FD simulation
        f0 = c0/source[:,2].min()# approximate frequency of wave
        DELTAX = DELTAY = 1/(f0*10)# target fine sampled deltas
        DELTAT = DELTAX / (4*np.sqrt(2)*c0)# target fine sampled deltas
        dx, dy, dt = int(np.ceil(deltax/DELTAX)), int(np.ceil(deltay/DELTAY)), int(np.ceil(deltat/DELTAT))# make sure deltas are a multiple of test deltas
        DELTAX, DELTAY, DELTAT = deltax/dx, deltay/dy, deltat/dt
        NX, NY, NSTEPS = batch_shape[0]*dx-(dx-1), batch_shape[1]*dy-(dy-1), batch_shape[2]*dt-(dt-1)

        # get starting wavefield
        xx,yy = np.meshgrid(np.linspace(xmin, xmax, NX), np.linspace(ymin, ymax, NY), indexing="ij")# (NX, NY)
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)# (n, 2)
        exp = np.exp
        p = np.expand_dims(source, axis=1)# (k, 1, 4)
        x = np.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)
        p0 = f.reshape((NX, NY))

        # get velocity model
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)# (n, 2)
        c = np.array(c_fn(all_params, x))
        if c.shape[0]>1: c = c.reshape((NX, NY))
        else: c = c*np.ones_like(xx)

        # add padded CPML boundary
        NPOINTS_PML = 10
        p0 = np.pad(p0, [(NPOINTS_PML,NPOINTS_PML),(NPOINTS_PML,NPOINTS_PML)], mode="edge")
        c =   np.pad(c, [(NPOINTS_PML,NPOINTS_PML),(NPOINTS_PML,NPOINTS_PML)], mode="edge")

        # run simulation
        logger.info(f'Running seismicCPML2D {(NX, NY, NSTEPS)}..')
        wavefields, _ = seismicCPML2D(
                    NX+2*NPOINTS_PML,
                    NY+2*NPOINTS_PML,
                    NSTEPS,
                    DELTAX,
                    DELTAY,
                    DELTAT,
                    NPOINTS_PML,
                    c,
                    np.ones((NX+2*NPOINTS_PML,NY+2*NPOINTS_PML)),
                    (p0.copy(),p0.copy()),
                    f0,
                    np.float32,
                    output_wavefields=True,
                    gather_is=None)

        # get croped, decimated, flattened wavefields
        wavefields = wavefields[:,NPOINTS_PML:-NPOINTS_PML,NPOINTS_PML:-NPOINTS_PML]
        wavefields = wavefields[::dt, ::dx, ::dy]
        wavefields = np.moveaxis(wavefields, 0, -1)
        assert wavefields.shape == batch_shape
        u = wavefields.reshape((-1, 1))

        return u

    @staticmethod
    def c_fn(all_params, x_batch):
        "Computes the velocity model"

        c0 = all_params["static"]["problem"]["c0"]
        return jnp.array([[c0]], dtype=float)# (1,1) scalar value


class WaveEquationGaussianVelocity3D(WaveEquationConstantVelocity3D):
    """Solves the time-dependent (2+1)D wave equation with gaussian mixture velocity
        d^2 u   d^2 u    1  d^2 u
        ----- + ----- - --- ----- = 0
        dx^2    dy^2    c^2 dt^2

        Boundary conditions:
        u(x,y,0) = amp * exp( -0.5 (||[x,y]-mu||/sd)^2 )
        du
        --(x,y,0) = 0
        dt
    """

    @staticmethod
    def init_params(c0=1, source=np.array([[0., 0., 0.2, 1.]]), mixture=np.array([[0.5, 0.5, 1., 0.2]])):

        static_params = {
            "dims":(1,3),
            "c0":c0,
            "c_fn":WaveEquationGaussianVelocity3D.c_fn,# velocity function
            "source":jnp.array(source),# location, width and amplitude of initial gaussian sources (k, 4)
            "mixture":jnp.array(mixture),# location, width and amplitude of gaussian pertubations in velocity model (l, 4)
            }
        return static_params, {}

    @staticmethod
    def c_fn(all_params, x_batch):
        "Computes the velocity model"

        c0, mixture = all_params["static"]["problem"]["c0"], all_params["static"]["problem"]["mixture"]
        x = x_batch[:,0:2]# (n, 2)
        exp = jnp.exp

        # get velocity model
        p = jnp.expand_dims(mixture, axis=1)# (l, 1, 4)
        x = jnp.expand_dims(x, axis=0)# (1, n, 2)
        f = (p[:,:,3:4]*exp(-0.5 * ((x-p[:,:,0:2])**2).sum(2, keepdims=True)/(p[:,:,2:3]**2))).sum(0)# (n, 1)
        c = c0 + f# (n, 1)
        return c




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from fbpinns.domains import RectangularDomainND

    np.random.seed(0)

    mixture=np.concatenate([
        np.random.uniform(-3, 3, (100,2)),# location
        0.4*np.ones((100,1)),# width
        0.3*np.random.uniform(-1, 1, (100,1)),# amplitude
        ], axis=1)

    source=np.array([# multiscale sources
        [0,0,0.1,1],
        [1,1,0.2,0.5],
        [-1.1,-0.5,0.4,0.25],
        ])

    # test wave equation
    for problem, kwargs in [(WaveEquationConstantVelocity3D, dict()),
                            (WaveEquationGaussianVelocity3D, dict(source=source, mixture=mixture))]:

        ps_ = problem.init_params(**kwargs)
        all_params = {"static":{"problem":ps_[0]}, "trainable":{"problem":ps_[1]}}

        batch_shape = (80,80,50)
        x_batch = RectangularDomainND._rectangle_samplerND(None, "grid", np.array([-3, -3, 0]), np.array([3, 3, 3]), batch_shape)

        plt.figure()
        c = np.array(problem.c_fn(all_params, x_batch))
        if c.shape[0]>1: c = c.reshape(batch_shape)
        else: c = c*np.ones(batch_shape)
        plt.imshow(c[:,:,0])
        plt.colorbar()
        plt.show()

        u = problem.exact_solution(all_params, x_batch, batch_shape).reshape(batch_shape)
        uc = np.zeros_like(x_batch)[:,0:1]
        uc = problem.constraining_fn(all_params, x_batch, uc).reshape(batch_shape)

        its = range(0,50,3)
        for u_ in [u, uc]:
            vmin, vmax = np.quantile(u, 0.05), np.quantile(u, 0.95)
            plt.figure(figsize=(2*len(its),5))
            for iplot,i in enumerate(its):
                plt.subplot(1,len(its),1+iplot)
                plt.imshow(u_[:,:,i], vmin=vmin, vmax=vmax)
            plt.show()
        plt.figure()
        plt.plot(u[40,40,:], label="u")
        plt.plot(uc[40,40,:], label="uc")
        t = np.linspace(0,1,50)
        plt.plot(np.tanh(2.5*t/(all_params["static"]["problem"]["source"][:,2].min()/all_params["static"]["problem"]["c0"]))**2)
        plt.legend()
        plt.show()



