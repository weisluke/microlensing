#include "mif.cuh"


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
    
    LIB_EXPORT MIF<dtype>* MIF_init()                                      {return new MIF<dtype>();}

    LIB_EXPORT void set_kappa_tot(MIF<dtype> *self, dtype val)             {self->kappa_tot            = val;}
    LIB_EXPORT void set_shear(MIF<dtype> *self, dtype val)                 {self->shear                = val;}
    LIB_EXPORT void set_kappa_star(MIF<dtype> *self, dtype val)            {self->kappa_star           = val;}
    LIB_EXPORT void set_theta_star(MIF<dtype> *self, dtype val)            {self->theta_star           = val;}
    LIB_EXPORT void set_mass_function(MIF<dtype> *self, const char* val)   {self->mass_function_str    = val;}
    LIB_EXPORT void set_m_solar(MIF<dtype> *self, dtype val)               {self->m_solar              = val;}
    LIB_EXPORT void set_m_lower(MIF<dtype> *self, dtype val)               {self->m_lower              = val;}
    LIB_EXPORT void set_m_upper(MIF<dtype> *self, dtype val)               {self->m_upper              = val;}
    LIB_EXPORT void set_light_loss(MIF<dtype> *self, dtype val)            {self->light_loss           = val;}
    LIB_EXPORT void set_rectangular(MIF<dtype> *self, int val)             {self->rectangular          = val;}
    LIB_EXPORT void set_approx(MIF<dtype> *self, int val)                  {self->approx               = val;}
    LIB_EXPORT void set_safety_scale(MIF<dtype> *self, dtype val)          {self->safety_scale         = val;}
    LIB_EXPORT void set_starfile(MIF<dtype> *self, const char* val)        {self->starfile             = val;}
    LIB_EXPORT void set_y1(MIF<dtype> *self, dtype val)                    {self->w0.re                = val;}
    LIB_EXPORT void set_y2(MIF<dtype> *self, dtype val)                    {self->w0.im                = val;}
    LIB_EXPORT void set_v1(MIF<dtype> *self, dtype val)                    {self->v.re                 = val;}
    LIB_EXPORT void set_v2(MIF<dtype> *self, dtype val)                    {self->v.im                 = val;}
    LIB_EXPORT void set_random_seed(MIF<dtype> *self, int val)             {self->random_seed          = val;}
    LIB_EXPORT void set_write_stars(MIF<dtype> *self, int val)             {self->write_stars          = val;}
    LIB_EXPORT void set_write_images(MIF<dtype> *self, int val)            {self->write_images         = val;}
    LIB_EXPORT void set_write_image_lines(MIF<dtype> *self, int val)       {self->write_image_lines    = val;}
    LIB_EXPORT void set_write_magnifications(MIF<dtype> *self, int val)    {self->write_magnifications = val;}
    LIB_EXPORT void set_outfile_prefix(MIF<dtype> *self, const char* val)  {self->outfile_prefix       = val;}

    LIB_EXPORT dtype get_kappa_tot(MIF<dtype> *self)                       {return self->kappa_tot;}
    LIB_EXPORT dtype get_shear(MIF<dtype> *self)                           {return self->shear;}
    LIB_EXPORT dtype get_kappa_star(MIF<dtype> *self)                      {return self->kappa_star;}
    LIB_EXPORT dtype get_theta_star(MIF<dtype> *self)                      {return self->theta_star;}
    LIB_EXPORT const char* get_mass_function(MIF<dtype> *self)             {return (self->mass_function_str).c_str();}
    LIB_EXPORT dtype get_m_solar(MIF<dtype> *self)                         {return self->m_solar;}
    LIB_EXPORT dtype get_m_lower(MIF<dtype> *self)                         {return self->m_lower;}
    LIB_EXPORT dtype get_m_upper(MIF<dtype> *self)                         {return self->m_upper;}
    LIB_EXPORT dtype get_light_loss(MIF<dtype> *self)                      {return self->light_loss;}
    LIB_EXPORT int get_rectangular(MIF<dtype> *self)                       {return self->rectangular;}
    LIB_EXPORT int get_approx(MIF<dtype> *self)                            {return self->approx;}
    LIB_EXPORT dtype get_safety_scale(MIF<dtype> *self)                    {return self->safety_scale;}
    LIB_EXPORT const char* get_starfile(MIF<dtype> *self)                  {return (self->starfile).c_str();}
    LIB_EXPORT dtype get_y1(MIF<dtype> *self)                              {return self->w0.re;}
    LIB_EXPORT dtype get_y2(MIF<dtype> *self)                              {return self->w0.im;}
    LIB_EXPORT dtype get_v1(MIF<dtype> *self)                              {return self->v.re;}
    LIB_EXPORT dtype get_v2(MIF<dtype> *self)                              {return self->v.im;}
    LIB_EXPORT int get_random_seed(MIF<dtype> *self)                       {return self->random_seed;}
    LIB_EXPORT int get_write_stars(MIF<dtype> *self)                       {return self->write_stars;}
    LIB_EXPORT int get_write_images(MIF<dtype> *self)                      {return self->write_images;}
    LIB_EXPORT int get_write_image_lines(MIF<dtype> *self)                 {return self->write_image_lines;}
    LIB_EXPORT int get_write_magnifications(MIF<dtype> *self)              {return self->write_magnifications;}
    LIB_EXPORT const char* get_outfile_prefix(MIF<dtype> *self)            {return (self->outfile_prefix).c_str();}

    LIB_EXPORT int get_num_stars(MIF<dtype> *self)                         {return self->get_num_stars();}
    LIB_EXPORT dtype get_corner_x1(MIF<dtype> *self)                       {return self->get_corner().re;}
    LIB_EXPORT dtype get_corner_x2(MIF<dtype> *self)                       {return self->get_corner().im;}
    LIB_EXPORT dtype* get_stars(MIF<dtype> *self)                          {return &(self->get_stars()[0].position.re);}
    LIB_EXPORT dtype* get_images(MIF<dtype> *self)                         {return &(self->get_images()[0].re);}
    LIB_EXPORT int get_num_images(MIF<dtype> *self)                        {return self->get_num_images();}
    LIB_EXPORT dtype* get_images_mags(MIF<dtype> *self)                    {return &(self->get_images_mags()[0].re);}
    LIB_EXPORT dtype* get_image_lines(MIF<dtype> *self)                    {return &(self->get_image_lines()[0].re);}
    LIB_EXPORT int get_num_image_lines(MIF<dtype> *self)                   {return self->get_num_image_lines();}
    LIB_EXPORT int* get_image_lines_lengths(MIF<dtype> *self)              {return self->get_image_lines_lengths();}
    LIB_EXPORT int get_total_image_lines_length(MIF<dtype> *self)          {return self->get_total_image_lines_length();}
    LIB_EXPORT dtype* get_source_lines(MIF<dtype> *self)                   {return &(self->get_source_lines()[0].re);}
    LIB_EXPORT dtype* get_image_lines_mags(MIF<dtype> *self)               {return self->get_image_lines_mags();}

    LIB_EXPORT bool run(MIF<dtype> *self, int verbose)                     {return self->run(verbose);}
    LIB_EXPORT bool save(MIF<dtype> *self, int verbose)                    {return self->save(verbose);}

    LIB_EXPORT void MIF_delete(MIF<dtype> *self)                           {delete self;}

}