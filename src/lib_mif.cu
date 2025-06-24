#include "mif.cuh"


#if defined(is_float) && !defined(is_double)
using dtype = float; //type to be used throughout this program. float or double
#elif !defined(is_float) && defined(is_double)
using dtype = double; //type to be used throughout this program. float or double
#else
#error "Error. One, and only one, of is_float or is_double must be defined"
#endif

extern "C" 
{
    
    MIF<dtype>* MIF_init()                                      {return new MIF<dtype>();}

    void set_kappa_tot(MIF<dtype> *self, dtype val)             {self->kappa_tot            = val;}
    void set_shear(MIF<dtype> *self, dtype val)                 {self->shear                = val;}
    void set_kappa_star(MIF<dtype> *self, dtype val)            {self->kappa_star           = val;}
    void set_theta_star(MIF<dtype> *self, dtype val)            {self->theta_star           = val;}
    void set_mass_function(MIF<dtype> *self, const char* val)   {self->mass_function_str    = val;}
    void set_m_solar(MIF<dtype> *self, dtype val)               {self->m_solar              = val;}
    void set_m_lower(MIF<dtype> *self, dtype val)               {self->m_lower              = val;}
    void set_m_upper(MIF<dtype> *self, dtype val)               {self->m_upper              = val;}
    void set_light_loss(MIF<dtype> *self, dtype val)            {self->light_loss           = val;}
    void set_rectangular(MIF<dtype> *self, int val)             {self->rectangular          = val;}
    void set_approx(MIF<dtype> *self, int val)                  {self->approx               = val;}
    void set_safety_scale(MIF<dtype> *self, dtype val)          {self->safety_scale         = val;}
    void set_starfile(MIF<dtype> *self, const char* val)        {self->starfile             = val;}
    void set_center_y1(MIF<dtype> *self, dtype val)             {self->center_y.re          = val;}
    void set_center_y2(MIF<dtype> *self, dtype val)             {self->center_y.im          = val;}
    void set_half_length_y1(MIF<dtype> *self, dtype val)        {self->half_length_y.re     = val;}
    void set_half_length_y2(MIF<dtype> *self, dtype val)        {self->half_length_y.im     = val;}
    void set_num_pixels_y1(MIF<dtype> *self, int val)           {self->num_pixels_y.re      = val;}
    void set_num_pixels_y2(MIF<dtype> *self, int val)           {self->num_pixels_y.im      = val;}
    void set_num_rays_y(MIF<dtype> *self, int val)              {self->num_rays_y           = val;}
    void set_random_seed(MIF<dtype> *self, int val)             {self->random_seed          = val;}
    void set_write_stars(MIF<dtype> *self, int val)             {self->write_stars          = val;}
    void set_write_maps(MIF<dtype> *self, int val)              {self->write_maps           = val;}
    void set_write_parities(MIF<dtype> *self, int val)          {self->write_parities       = val;}
    void set_write_histograms(MIF<dtype> *self, int val)        {self->write_histograms     = val;}
    void set_outfile_prefix(MIF<dtype> *self, const char* val)  {self->outfile_prefix       = val;}

    dtype get_kappa_tot(MIF<dtype> *self)                       {return self->kappa_tot;}
    dtype get_shear(MIF<dtype> *self)                           {return self->shear;}
    dtype get_kappa_star(MIF<dtype> *self)                      {return self->kappa_star;}
    dtype get_theta_star(MIF<dtype> *self)                      {return self->theta_star;}
    const char* get_mass_function(MIF<dtype> *self)             {return (self->mass_function_str).c_str();}
    dtype get_m_solar(MIF<dtype> *self)                         {return self->m_solar;}
    dtype get_m_lower(MIF<dtype> *self)                         {return self->m_lower;}
    dtype get_m_upper(MIF<dtype> *self)                         {return self->m_upper;}
    dtype get_light_loss(MIF<dtype> *self)                      {return self->light_loss;}
    int get_rectangular(MIF<dtype> *self)                       {return self->rectangular;}
    int get_approx(MIF<dtype> *self)                            {return self->approx;}
    dtype get_safety_scale(MIF<dtype> *self)                    {return self->safety_scale;}
    const char* get_starfile(MIF<dtype> *self)                  {return (self->starfile).c_str();}
    dtype get_center_y1(MIF<dtype> *self)                       {return self->center_y.re;}
    dtype get_center_y2(MIF<dtype> *self)                       {return self->center_y.im;}
    dtype get_half_length_y1(MIF<dtype> *self)                  {return self->half_length_y.re;}
    dtype get_half_length_y2(MIF<dtype> *self)                  {return self->half_length_y.im;}
    int get_num_pixels_y1(MIF<dtype> *self)                     {return self->num_pixels_y.re;}
    int get_num_pixels_y2(MIF<dtype> *self)                     {return self->num_pixels_y.im;}
    int get_num_rays_y(MIF<dtype> *self)                        {return self->num_rays_y;}
    int get_random_seed(MIF<dtype> *self)                       {return self->random_seed;}
    int get_write_stars(MIF<dtype> *self)                       {return self->write_stars;}
    int get_write_maps(MIF<dtype> *self)                        {return self->write_maps;}
    int get_write_parities(MIF<dtype> *self)                    {return self->write_parities;}
    int get_write_histograms(MIF<dtype> *self)                  {return self->write_histograms;}
    const char* get_outfile_prefix(MIF<dtype> *self)            {return (self->outfile_prefix).c_str();}

    int get_num_stars(MIF<dtype> *self)                         {return self->get_num_stars();}
    dtype get_corner_x1(MIF<dtype> *self)                       {return self->get_corner().re;}
    dtype get_corner_x2(MIF<dtype> *self)                       {return self->get_corner().im;}
    dtype* get_stars(MIF<dtype> *self)                          {return &(self->get_stars()[0].position.re);}
    dtype* get_pixels(MIF<dtype> *self)                         {return self->get_pixels();}
    dtype* get_pixels_minima(MIF<dtype> *self)                  {return self->get_pixels_minima();}
    dtype* get_pixels_saddles(MIF<dtype> *self)                 {return self->get_pixels_saddles();}
    double get_t_shoot_cells(MIF<dtype> *self)                  {return self->get_t_shoot_cells();}

    bool run(MIF<dtype> *self, int verbose)                     {return self->run(verbose);}
    bool save(MIF<dtype> *self, int verbose)                    {return self->save(verbose);}

    void MIF_delete(MIF<dtype> *self)                           {delete self;}

}