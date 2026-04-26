#include "ccf.cuh"


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
    
    LIB_EXPORT CCF<dtype>* CCF_init()                                      {return new CCF<dtype>();}

    LIB_EXPORT void set_kappa_tot(CCF<dtype> *self, dtype val)             {self->kappa_tot                = val;}
    LIB_EXPORT void set_shear(CCF<dtype> *self, dtype val)                 {self->shear                    = val;}
    LIB_EXPORT void set_kappa_star(CCF<dtype> *self, dtype val)            {self->kappa_star               = val;}
    LIB_EXPORT void set_theta_star(CCF<dtype> *self, dtype val)            {self->theta_star               = val;}
    LIB_EXPORT void set_mass_function(CCF<dtype> *self, const char* val)   {self->mass_function_str        = val;}
    LIB_EXPORT void set_m_solar(CCF<dtype> *self, dtype val)               {self->m_solar                  = val;}
    LIB_EXPORT void set_m_lower(CCF<dtype> *self, dtype val)               {self->m_lower                  = val;}
    LIB_EXPORT void set_m_upper(CCF<dtype> *self, dtype val)               {self->m_upper                  = val;}
    LIB_EXPORT void set_rectangular(CCF<dtype> *self, int val)             {self->rectangular              = val;}
    LIB_EXPORT void set_approx(CCF<dtype> *self, int val)                  {self->approx                   = val;}
    LIB_EXPORT void set_safety_scale(CCF<dtype> *self, dtype val)          {self->safety_scale             = val;}
    LIB_EXPORT void set_num_stars(CCF<dtype> *self, int val)               {self->num_stars                = val;}
    LIB_EXPORT void set_starfile(CCF<dtype> *self, const char* val)        {self->starfile                 = val;}
    LIB_EXPORT void set_num_phi(CCF<dtype> *self, int val)                 {self->num_phi                  = val;}
    LIB_EXPORT void set_num_branches(CCF<dtype> *self, int val)            {self->num_branches             = val;}
    LIB_EXPORT void set_random_seed(CCF<dtype> *self, int val)             {self->random_seed              = val;}
    LIB_EXPORT void set_write_stars(CCF<dtype> *self, int val)             {self->write_stars              = val;}
    LIB_EXPORT void set_write_critical_curves(CCF<dtype> *self, int val)   {self->write_critical_curves    = val;}
    LIB_EXPORT void set_write_caustics(CCF<dtype> *self, int val)          {self->write_caustics           = val;}
    LIB_EXPORT void set_write_mu_length_scales(CCF<dtype> *self, int val)  {self->write_mu_length_scales   = val;}
    LIB_EXPORT void set_outfile_prefix(CCF<dtype> *self, const char* val)  {self->outfile_prefix           = val;}

    LIB_EXPORT dtype get_kappa_tot(CCF<dtype> *self)                       {return self->kappa_tot;}
    LIB_EXPORT dtype get_shear(CCF<dtype> *self)                           {return self->shear;}
    LIB_EXPORT dtype get_kappa_star(CCF<dtype> *self)                      {return self->kappa_star;}
    LIB_EXPORT dtype get_theta_star(CCF<dtype> *self)                      {return self->theta_star;}
    LIB_EXPORT const char* get_mass_function(CCF<dtype> *self)             {return (self->mass_function_str).c_str();}
    LIB_EXPORT dtype get_m_solar(CCF<dtype> *self)                         {return self->m_solar;}
    LIB_EXPORT dtype get_m_lower(CCF<dtype> *self)                         {return self->m_lower;}
    LIB_EXPORT dtype get_m_upper(CCF<dtype> *self)                         {return self->m_upper;}
    LIB_EXPORT int get_rectangular(CCF<dtype> *self)                       {return self->rectangular;}
    LIB_EXPORT int get_approx(CCF<dtype> *self)                            {return self->approx;}
    LIB_EXPORT dtype get_safety_scale(CCF<dtype> *self)                    {return self->safety_scale;}
    LIB_EXPORT int get_num_stars(CCF<dtype> *self)                         {return self->num_stars;}
    LIB_EXPORT const char* get_starfile(CCF<dtype> *self)                  {return (self->starfile).c_str();}
    LIB_EXPORT int get_num_phi(CCF<dtype> *self)                           {return self->num_phi;}
    LIB_EXPORT int get_num_branches(CCF<dtype> *self)                      {return self->num_branches;}
    LIB_EXPORT int get_random_seed(CCF<dtype> *self)                       {return self->random_seed;}
    LIB_EXPORT int get_write_stars(CCF<dtype> *self)                       {return self->write_stars;}
    LIB_EXPORT int get_write_critical_curves(CCF<dtype> *self)             {return self->write_critical_curves;}
    LIB_EXPORT int get_write_caustics(CCF<dtype> *self)                    {return self->write_caustics;}
    LIB_EXPORT int get_write_mu_length_scales(CCF<dtype> *self)            {return self->write_mu_length_scales;}
    LIB_EXPORT const char* get_outfile_prefix(CCF<dtype> *self)            {return (self->outfile_prefix).c_str();}

    LIB_EXPORT int get_num_roots(CCF<dtype> *self)                         {return self->get_num_roots();}
    LIB_EXPORT dtype get_corner_x1(CCF<dtype> *self)                       {return self->get_corner().re;}
    LIB_EXPORT dtype get_corner_x2(CCF<dtype> *self)                       {return self->get_corner().im;}
    LIB_EXPORT dtype* get_stars(CCF<dtype> *self)                          {return &(self->get_stars()[0].position.re);}
    LIB_EXPORT dtype* get_critical_curves(CCF<dtype> *self)                {return &(self->get_critical_curves()[0].re);}
    LIB_EXPORT dtype* get_caustics(CCF<dtype> *self)                       {return &(self->get_caustics()[0].re);}
    LIB_EXPORT dtype* get_mu_length_scales(CCF<dtype> *self)               {return self->get_mu_length_scales();}
    LIB_EXPORT double get_t_ccs(CCF<dtype> *self)                          {return self->get_t_ccs();}

    LIB_EXPORT bool run(CCF<dtype> *self, int verbose)                     {return self->run(verbose);}
    LIB_EXPORT bool save(CCF<dtype> *self, int verbose)                    {return self->save(verbose);}

    LIB_EXPORT void CCF_delete(CCF<dtype> *self)                           {delete self;}

}