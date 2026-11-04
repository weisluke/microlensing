#include "ipm.cuh"


#if defined(is_float) && !defined(is_double)
using dtype = float; //type to be used throughout this program. float or double
#elif !defined(is_float) && defined(is_double)
using dtype = double; //type to be used throughout this program. float or double
#else
#error "Error. One, and only one, of is_float or is_double must be defined"
#endif

#if defined(_WIN32) || defined(_WIN64)
    #define LIB_EXPORT __declspec(dllexport)
#else
    #define LIB_EXPORT
#endif

extern "C" 
{
    
    LIB_EXPORT IPM<dtype>* IPM_init()                                      {return new IPM<dtype>();}

    LIB_EXPORT void set_kappa_tot(IPM<dtype> *self, dtype val)             {self->kappa_tot            = val;}
    LIB_EXPORT void set_shear(IPM<dtype> *self, dtype val)                 {self->shear                = val;}
    LIB_EXPORT void set_kappa_star(IPM<dtype> *self, dtype val)            {self->kappa_star           = val;}
    LIB_EXPORT void set_theta_star(IPM<dtype> *self, dtype val)            {self->theta_star           = val;}
    LIB_EXPORT void set_mass_function(IPM<dtype> *self, const char* val)   {self->mass_function_str    = val;}
    LIB_EXPORT void set_m_solar(IPM<dtype> *self, dtype val)               {self->m_solar              = val;}
    LIB_EXPORT void set_m_lower(IPM<dtype> *self, dtype val)               {self->m_lower              = val;}
    LIB_EXPORT void set_m_upper(IPM<dtype> *self, dtype val)               {self->m_upper              = val;}
    LIB_EXPORT void set_light_loss(IPM<dtype> *self, dtype val)            {self->light_loss           = val;}
    LIB_EXPORT void set_rectangular(IPM<dtype> *self, int val)             {self->rectangular          = val;}
    LIB_EXPORT void set_approx(IPM<dtype> *self, int val)                  {self->approx               = val;}
    LIB_EXPORT void set_safety_scale(IPM<dtype> *self, dtype val)          {self->safety_scale         = val;}
    LIB_EXPORT void set_starfile(IPM<dtype> *self, const char* val)        {self->starfile             = val;}
    LIB_EXPORT void set_center_y1(IPM<dtype> *self, dtype val)             {self->center_y.re          = val;}
    LIB_EXPORT void set_center_y2(IPM<dtype> *self, dtype val)             {self->center_y.im          = val;}
    LIB_EXPORT void set_half_length_y1(IPM<dtype> *self, dtype val)        {self->half_length_y.re     = val;}
    LIB_EXPORT void set_half_length_y2(IPM<dtype> *self, dtype val)        {self->half_length_y.im     = val;}
    LIB_EXPORT void set_num_pixels_y1(IPM<dtype> *self, int val)           {self->num_pixels_y.re      = val;}
    LIB_EXPORT void set_num_pixels_y2(IPM<dtype> *self, int val)           {self->num_pixels_y.im      = val;}
    LIB_EXPORT void set_num_rays_y(IPM<dtype> *self, int val)              {self->num_rays_y           = val;}
    LIB_EXPORT void set_random_seed(IPM<dtype> *self, int val)             {self->random_seed          = val;}
    LIB_EXPORT void set_write_stars(IPM<dtype> *self, int val)             {self->write_stars          = val;}
    LIB_EXPORT void set_write_maps(IPM<dtype> *self, int val)              {self->write_maps           = val;}
    LIB_EXPORT void set_write_parities(IPM<dtype> *self, int val)          {self->write_parities       = val;}
    LIB_EXPORT void set_write_histograms(IPM<dtype> *self, int val)        {self->write_histograms     = val;}
    LIB_EXPORT void set_outfile_prefix(IPM<dtype> *self, const char* val)  {self->outfile_prefix       = val;}

    LIB_EXPORT dtype get_kappa_tot(IPM<dtype> *self)                       {return self->kappa_tot;}
    LIB_EXPORT dtype get_shear(IPM<dtype> *self)                           {return self->shear;}
    LIB_EXPORT dtype get_kappa_star(IPM<dtype> *self)                      {return self->kappa_star;}
    LIB_EXPORT dtype get_theta_star(IPM<dtype> *self)                      {return self->theta_star;}
    LIB_EXPORT const char* get_mass_function(IPM<dtype> *self)             {return (self->mass_function_str).c_str();}
    LIB_EXPORT dtype get_m_solar(IPM<dtype> *self)                         {return self->m_solar;}
    LIB_EXPORT dtype get_m_lower(IPM<dtype> *self)                         {return self->m_lower;}
    LIB_EXPORT dtype get_m_upper(IPM<dtype> *self)                         {return self->m_upper;}
    LIB_EXPORT dtype get_light_loss(IPM<dtype> *self)                      {return self->light_loss;}
    LIB_EXPORT int get_rectangular(IPM<dtype> *self)                       {return self->rectangular;}
    LIB_EXPORT int get_approx(IPM<dtype> *self)                            {return self->approx;}
    LIB_EXPORT dtype get_safety_scale(IPM<dtype> *self)                    {return self->safety_scale;}
    LIB_EXPORT const char* get_starfile(IPM<dtype> *self)                  {return (self->starfile).c_str();}
    LIB_EXPORT dtype get_center_y1(IPM<dtype> *self)                       {return self->center_y.re;}
    LIB_EXPORT dtype get_center_y2(IPM<dtype> *self)                       {return self->center_y.im;}
    LIB_EXPORT dtype get_half_length_y1(IPM<dtype> *self)                  {return self->half_length_y.re;}
    LIB_EXPORT dtype get_half_length_y2(IPM<dtype> *self)                  {return self->half_length_y.im;}
    LIB_EXPORT int get_num_pixels_y1(IPM<dtype> *self)                     {return self->num_pixels_y.re;}
    LIB_EXPORT int get_num_pixels_y2(IPM<dtype> *self)                     {return self->num_pixels_y.im;}
    LIB_EXPORT int get_num_rays_y(IPM<dtype> *self)                        {return self->num_rays_y;}
    LIB_EXPORT int get_random_seed(IPM<dtype> *self)                       {return self->random_seed;}
    LIB_EXPORT int get_write_stars(IPM<dtype> *self)                       {return self->write_stars;}
    LIB_EXPORT int get_write_maps(IPM<dtype> *self)                        {return self->write_maps;}
    LIB_EXPORT int get_write_parities(IPM<dtype> *self)                    {return self->write_parities;}
    LIB_EXPORT int get_write_histograms(IPM<dtype> *self)                  {return self->write_histograms;}
    LIB_EXPORT const char* get_outfile_prefix(IPM<dtype> *self)            {return (self->outfile_prefix).c_str();}

    LIB_EXPORT int get_num_stars(IPM<dtype> *self)                         {return self->get_num_stars();}
    LIB_EXPORT dtype get_corner_x1(IPM<dtype> *self)                       {return self->get_corner().re;}
    LIB_EXPORT dtype get_corner_x2(IPM<dtype> *self)                       {return self->get_corner().im;}
    LIB_EXPORT dtype* get_stars(IPM<dtype> *self)                          {return &(self->get_stars()[0].position.re);}
    LIB_EXPORT dtype* get_pixels(IPM<dtype> *self)                         {return self->get_pixels();}
    LIB_EXPORT dtype* get_pixels_minima(IPM<dtype> *self)                  {return self->get_pixels_minima();}
    LIB_EXPORT dtype* get_pixels_saddles(IPM<dtype> *self)                 {return self->get_pixels_saddles();}
    LIB_EXPORT bool copy_pixels(IPM<dtype> *self, dtype* dst)              {return self->copy_pixels(dst);}
    LIB_EXPORT bool copy_pixels_minima(IPM<dtype> *self, dtype* dst)       {return self->copy_pixels_minima(dst);}
    LIB_EXPORT bool copy_pixels_saddles(IPM<dtype> *self, dtype* dst)      {return self->copy_pixels_saddles(dst);}
    LIB_EXPORT double get_t_shoot_cells(IPM<dtype> *self)                  {return self->get_t_shoot_cells();}

    LIB_EXPORT bool run(IPM<dtype> *self, int verbose)                     {return self->run(verbose);}
    LIB_EXPORT bool save(IPM<dtype> *self, int verbose)                    {return self->save(verbose);}

    LIB_EXPORT bool set_worker(IPM<dtype> *self, int device)
    {
        int count = 0;
        cudaGetDeviceCount(&count);
        if (cuda_error("cudaGetDeviceCount", false, __FILE__, __LINE__)) return false;
        if (device < 0 || device >= count) return false;
        self->device = device;
        self->reuse_memory = 1;
        return self->create_stream();
    }
    LIB_EXPORT bool release_memory(IPM<dtype> *self)                       {return self->release_memory();}
    LIB_EXPORT void reset_inputs(IPM<dtype> *self)                         {self->reset_inputs();}
    LIB_EXPORT bool estimate_memory(IPM<dtype> *self, unsigned long long* out4)
    {
        return self->estimate_memory(out4[0], out4[1], out4[2], out4[3]);
    }
    LIB_EXPORT void get_allocated_bytes(IPM<dtype> *self, unsigned long long* out3)
    {
        self->allocated_bytes(out3[0], out3[1], out3[2]);
    }
    LIB_EXPORT int get_perimeter_deposit_used(IPM<dtype> *self)            {return self->get_perimeter_deposit_used();}
    LIB_EXPORT IPM<dtype>* IPM_snapshot(IPM<dtype> *self)                  {return self->snapshot();}
    LIB_EXPORT bool save_host(IPM<dtype> *self, dtype* pixels, dtype* minima, dtype* saddles, int verbose)
    {
        return self->save_host(pixels, minima, saddles, verbose);
    }
    //util.cuh already declares a C++ take_last_cuda_error, so the C symbol of that name is given to another function
    #if defined(_WIN32) || defined(_WIN64)
        #pragma comment(linker, "/export:take_last_cuda_error=export_take_last_cuda_error")
        LIB_EXPORT int export_take_last_cuda_error()                       {return take_last_cuda_error();}
    #else
        LIB_EXPORT int export_take_last_cuda_error() __asm__("take_last_cuda_error");
        LIB_EXPORT int export_take_last_cuda_error()                       {return take_last_cuda_error();}
    #endif
    LIB_EXPORT int get_device_count()
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess)
        {
            cudaGetLastError();
            return 0;
        }
        return count;
    }
    LIB_EXPORT bool get_device_info(int device, unsigned long long* out2, int* concurrent_managed)
    {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaSetDevice(device) != cudaSuccess) {cudaGetLastError(); return false;}
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {cudaGetLastError(); return false;}
        if (cudaDeviceGetAttribute(concurrent_managed, cudaDevAttrConcurrentManagedAccess, device) != cudaSuccess) {cudaGetLastError(); return false;}
        out2[0] = free_bytes;
        out2[1] = total_bytes;
        return true;
    }
    LIB_EXPORT int check_device(int device)
    {
        cudaError_t err = cudaSetDevice(device);
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err != cudaSuccess) cudaGetLastError();
        return static_cast<int>(err);
    }

    LIB_EXPORT void IPM_delete(IPM<dtype> *self)                           {delete self;}

}