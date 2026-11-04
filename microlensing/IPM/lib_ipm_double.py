import platform
import ctypes
from importlib.resources import files


if platform.system() == 'Linux' and platform.architecture()[0] == '64bit':
    lib = ctypes.CDLL(files('microlensing.lib').joinpath('lib_ipm_double.so'))
elif platform.system() == 'Windows' and platform.architecture()[0] == '64bit':
    lib = ctypes.CDLL(files('microlensing.lib').joinpath('lib_ipm_double.dll'))
else:
    raise FileNotFoundError("IPM library only available for Linux and Windows 64bit platforms")

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
# the copy exports are missing from libraries built before them (such as the Windows DLLs); ipm.py then uses get_pixels
if hasattr(lib, "copy_pixels"):
    lib.copy_pixels.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.copy_pixels.restype = ctypes.c_bool
    lib.copy_pixels_minima.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.copy_pixels_minima.restype = ctypes.c_bool
    lib.copy_pixels_saddles.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.copy_pixels_saddles.restype = ctypes.c_bool
lib.get_t_shoot_cells.argtypes = [ctypes.c_void_p]
lib.get_t_shoot_cells.restype = ctypes.c_double

# the worker exports are missing from libraries built before them (such as the Windows DLLs)
if hasattr(lib, "set_worker"):
    lib.set_worker.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.set_worker.restype = ctypes.c_bool
    lib.release_memory.argtypes = [ctypes.c_void_p]
    lib.release_memory.restype = ctypes.c_bool
    lib.reset_inputs.argtypes = [ctypes.c_void_p]
    lib.reset_inputs.restype = None
    lib.estimate_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
    lib.estimate_memory.restype = ctypes.c_bool
    lib.get_allocated_bytes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
    lib.get_allocated_bytes.restype = None
    lib.get_perimeter_deposit_used.argtypes = [ctypes.c_void_p]
    lib.get_perimeter_deposit_used.restype = ctypes.c_int
    lib.IPM_snapshot.argtypes = [ctypes.c_void_p]
    lib.IPM_snapshot.restype = ctypes.c_void_p
    lib.save_host.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.save_host.restype = ctypes.c_bool
    lib.take_last_cuda_error.argtypes = []
    lib.take_last_cuda_error.restype = ctypes.c_int
    lib.get_device_count.argtypes = []
    lib.get_device_count.restype = ctypes.c_int
    lib.get_device_info.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int)]
    lib.get_device_info.restype = ctypes.c_bool
    lib.check_device.argtypes = [ctypes.c_int]
    lib.check_device.restype = ctypes.c_int

lib.run.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.run.restype = ctypes.c_bool
lib.save.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.save.restype = ctypes.c_bool

lib.IPM_delete.argtypes = [ctypes.c_void_p]
lib.IPM_delete.restype = None
