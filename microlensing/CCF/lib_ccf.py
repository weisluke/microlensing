import platform
import ctypes
from importlib.resources import files


if platform.system() == 'Linux' and platform.architecture()[0] == '64bit':
    lib = ctypes.CDLL(files('microlensing.lib').joinpath('lib_ccf.so'))
elif platform.system() == 'Windows' and platform.architecture()[0] == '64bit':
    lib = ctypes.CDLL(files('microlensing.lib').joinpath('lib_ccf.dll'))
else:
    raise FileNotFoundError("CCF library only available for Linux and Windows 64bit platforms")

lib.CCF_init.argtypes = []
lib.CCF_init.restype = ctypes.c_void_p

lib.set_kappa_tot.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_kappa_tot.restype = ctypes.c_void_p
lib.set_shear.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_shear.restype = ctypes.c_void_p
lib.set_kappa_star.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_kappa_star.restype = ctypes.c_void_p
lib.set_theta_star.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_theta_star.restype = ctypes.c_void_p
lib.set_mass_function.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.set_mass_function.restype = ctypes.c_void_p
lib.set_m_solar.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_m_solar.restype = ctypes.c_void_p
lib.set_m_lower.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_m_lower.restype = ctypes.c_void_p
lib.set_m_upper.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_m_upper.restype = ctypes.c_void_p
lib.set_rectangular.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_rectangular.restype = ctypes.c_void_p
lib.set_approx.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_approx.restype = ctypes.c_void_p
lib.set_safety_scale.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_safety_scale.restype = ctypes.c_void_p
lib.set_num_stars.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_num_stars.restype = ctypes.c_void_p
lib.set_starfile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.set_starfile.restype = ctypes.c_void_p
lib.set_num_phi.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_num_phi.restype = ctypes.c_void_p
lib.set_num_branches.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_num_branches.restype = ctypes.c_void_p
lib.set_random_seed.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_random_seed.restype = ctypes.c_void_p
lib.set_write_stars.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_stars.restype = ctypes.c_void_p
lib.set_write_critical_curves.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_critical_curves.restype = ctypes.c_void_p
lib.set_write_caustics.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_caustics.restype = ctypes.c_void_p
lib.set_write_mu_length_scales.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_mu_length_scales.restype = ctypes.c_void_p
lib.set_outfile_prefix.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.set_outfile_prefix.restype = ctypes.c_void_p

lib.get_kappa_tot.argtypes = [ctypes.c_void_p]
lib.get_kappa_tot.restype = ctypes.c_double
lib.get_shear.argtypes = [ctypes.c_void_p]
lib.get_shear.restype = ctypes.c_double
lib.get_kappa_star.argtypes = [ctypes.c_void_p]
lib.get_kappa_star.restype = ctypes.c_double
lib.get_theta_star.argtypes = [ctypes.c_void_p]
lib.get_theta_star.restype = ctypes.c_double
lib.get_mass_function.argtypes = [ctypes.c_void_p]
lib.get_mass_function.restype = ctypes.c_char_p
lib.get_m_solar.argtypes = [ctypes.c_void_p]
lib.get_m_solar.restype = ctypes.c_double
lib.get_m_lower.argtypes = [ctypes.c_void_p]
lib.get_m_lower.restype = ctypes.c_double
lib.get_m_upper.argtypes = [ctypes.c_void_p]
lib.get_m_upper.restype = ctypes.c_double
lib.get_rectangular.argtypes = [ctypes.c_void_p]
lib.get_rectangular.restype = ctypes.c_int
lib.get_approx.argtypes = [ctypes.c_void_p]
lib.get_approx.restype = ctypes.c_int
lib.get_safety_scale.argtypes = [ctypes.c_void_p]
lib.get_safety_scale.restype = ctypes.c_double
lib.get_num_stars.argtypes = [ctypes.c_void_p]
lib.get_num_stars.restype = ctypes.c_int
lib.get_starfile.argtypes = [ctypes.c_void_p]
lib.get_starfile.restype = ctypes.c_char_p
lib.get_num_phi.argtypes = [ctypes.c_void_p]
lib.get_num_phi.restype = ctypes.c_int
lib.get_num_branches.argtypes = [ctypes.c_void_p]
lib.get_num_branches.restype = ctypes.c_int
lib.get_random_seed.argtypes = [ctypes.c_void_p]
lib.get_random_seed.restype = ctypes.c_int
lib.get_write_stars.argtypes = [ctypes.c_void_p]
lib.get_write_stars.restype = ctypes.c_int
lib.get_write_critical_curves.argtypes = [ctypes.c_void_p]
lib.get_write_critical_curves.restype = ctypes.c_int
lib.get_write_caustics.argtypes = [ctypes.c_void_p]
lib.get_write_caustics.restype = ctypes.c_int
lib.get_write_mu_length_scales.argtypes = [ctypes.c_void_p]
lib.get_write_mu_length_scales.restype = ctypes.c_int
lib.get_outfile_prefix.argtypes = [ctypes.c_void_p]
lib.get_outfile_prefix.restype = ctypes.c_char_p

lib.get_num_roots.argtypes = [ctypes.c_void_p]
lib.get_num_roots.restype = ctypes.c_int
lib.get_corner_x1.argtypes = [ctypes.c_void_p]
lib.get_corner_x1.restype = ctypes.c_double
lib.get_corner_x2.argtypes = [ctypes.c_void_p]
lib.get_corner_x2.restype = ctypes.c_double
lib.get_stars.argtypes = [ctypes.c_void_p]
lib.get_stars.restype = ctypes.POINTER(ctypes.c_double)
lib.get_critical_curves.argtypes = [ctypes.c_void_p]
lib.get_critical_curves.restype = ctypes.POINTER(ctypes.c_double)
lib.get_caustics.argtypes = [ctypes.c_void_p]
lib.get_caustics.restype = ctypes.POINTER(ctypes.c_double)
lib.get_mu_length_scales.argtypes = [ctypes.c_void_p]
lib.get_mu_length_scales.restype = ctypes.POINTER(ctypes.c_double)
lib.get_t_ccs.argtypes = [ctypes.c_void_p]
lib.get_t_ccs.restype = ctypes.c_double

lib.run.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.run.restype = ctypes.c_bool
lib.save.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.save.restype = ctypes.c_bool

lib.CCF_delete.argtypes = [ctypes.c_void_p]
lib.CCF_delete.restype = None
