from sys import platform
import ctypes
from importlib.resources import files


if platform == 'linux':
    lib = ctypes.CDLL(files('microlensing.lib').joinpath('lib_ipm_double.so'))
else:
    raise FileNotFoundError("IPM library for non-Linux platforms not yet available")

lib.IPM_init.argtypes = []
lib.IPM_init.restype = ctypes.c_void_p

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
lib.set_light_loss.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_light_loss.restype = ctypes.c_void_p
lib.set_rectangular.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_rectangular.restype = ctypes.c_void_p
lib.set_approx.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_approx.restype = ctypes.c_void_p
lib.set_safety_scale.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_safety_scale.restype = ctypes.c_void_p
lib.set_starfile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.set_starfile.restype = ctypes.c_void_p
lib.set_center_y1.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_center_y1.restype = ctypes.c_void_p
lib.set_center_y2.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_center_y2.restype = ctypes.c_void_p
lib.set_half_length_y1.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_half_length_y1.restype = ctypes.c_void_p
lib.set_half_length_y2.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.set_half_length_y2.restype = ctypes.c_void_p
lib.set_num_pixels_y1.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_num_pixels_y1.restype = ctypes.c_void_p
lib.set_num_pixels_y2.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_num_pixels_y2.restype = ctypes.c_void_p
lib.set_num_rays_y.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_num_rays_y.restype = ctypes.c_void_p
lib.set_random_seed.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_random_seed.restype = ctypes.c_void_p
lib.set_write_stars.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_stars.restype = ctypes.c_void_p
lib.set_write_maps.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_maps.restype = ctypes.c_void_p
lib.set_write_parities.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_parities.restype = ctypes.c_void_p
lib.set_write_histograms.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.set_write_histograms.restype = ctypes.c_void_p
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
lib.get_light_loss.argtypes = [ctypes.c_void_p]
lib.get_light_loss.restype = ctypes.c_double
lib.get_rectangular.argtypes = [ctypes.c_void_p]
lib.get_rectangular.restype = ctypes.c_int
lib.get_approx.argtypes = [ctypes.c_void_p]
lib.get_approx.restype = ctypes.c_int
lib.get_safety_scale.argtypes = [ctypes.c_void_p]
lib.get_safety_scale.restype = ctypes.c_double
lib.get_starfile.argtypes = [ctypes.c_void_p]
lib.get_starfile.restype = ctypes.c_char_p
lib.get_center_y1.argtypes = [ctypes.c_void_p]
lib.get_center_y1.restype = ctypes.c_double
lib.get_center_y2.argtypes = [ctypes.c_void_p]
lib.get_center_y2.restype = ctypes.c_double
lib.get_half_length_y1.argtypes = [ctypes.c_void_p]
lib.get_half_length_y1.restype = ctypes.c_double
lib.get_half_length_y2.argtypes = [ctypes.c_void_p]
lib.get_half_length_y2.restype = ctypes.c_double
lib.get_num_pixels_y1.argtypes = [ctypes.c_void_p]
lib.get_num_pixels_y1.restype = ctypes.c_int
lib.get_num_pixels_y2.argtypes = [ctypes.c_void_p]
lib.get_num_pixels_y2.restype = ctypes.c_int
lib.get_num_rays_y.argtypes = [ctypes.c_void_p]
lib.get_num_rays_y.restype = ctypes.c_int
lib.get_random_seed.argtypes = [ctypes.c_void_p]
lib.get_random_seed.restype = ctypes.c_int
lib.get_write_stars.argtypes = [ctypes.c_void_p]
lib.get_write_stars.restype = ctypes.c_int
lib.get_write_maps.argtypes = [ctypes.c_void_p]
lib.get_write_maps.restype = ctypes.c_int
lib.get_write_parities.argtypes = [ctypes.c_void_p]
lib.get_write_parities.restype = ctypes.c_int
lib.get_write_histograms.argtypes = [ctypes.c_void_p]
lib.get_write_histograms.restype = ctypes.c_int
lib.get_outfile_prefix.argtypes = [ctypes.c_void_p]
lib.get_outfile_prefix.restype = ctypes.c_char_p

lib.get_num_stars.argtypes = [ctypes.c_void_p]
lib.get_num_stars.restype = ctypes.c_int
lib.get_corner_x1.argtypes = [ctypes.c_void_p]
lib.get_corner_x1.restype = ctypes.c_double
lib.get_corner_x2.argtypes = [ctypes.c_void_p]
lib.get_corner_x2.restype = ctypes.c_double
lib.get_stars.argtypes = [ctypes.c_void_p]
lib.get_stars.restype = ctypes.POINTER(ctypes.c_double)
lib.get_pixels.argtypes = [ctypes.c_void_p]
lib.get_pixels.restype = ctypes.POINTER(ctypes.c_double)
lib.get_pixels_minima.argtypes = [ctypes.c_void_p]
lib.get_pixels_minima.restype = ctypes.POINTER(ctypes.c_double)
lib.get_pixels_saddles.argtypes = [ctypes.c_void_p]
lib.get_pixels_saddles.restype = ctypes.POINTER(ctypes.c_double)

lib.run.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.run.restype = ctypes.c_bool
lib.save.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.save.restype = ctypes.c_bool

lib.IPM_delete.argtypes = [ctypes.c_void_p]
lib.IPM_delete.restype = None
