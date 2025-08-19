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
    void set_y1(MIF<dtype> *self, dtype val)                    {self->w0.re                = val;}
    void set_y2(MIF<dtype> *self, dtype val)                    {self->w0.im                = val;}
    void set_v1(MIF<dtype> *self, dtype val)                    {self->v.re                 = val;}
    void set_v2(MIF<dtype> *self, dtype val)                    {self->v.im                 = val;}
    void set_random_seed(MIF<dtype> *self, int val)             {self->random_seed          = val;}
    void set_write_stars(MIF<dtype> *self, int val)             {self->write_stars          = val;}
    void set_write_images(MIF<dtype> *self, int val)            {self->write_images         = val;}
    void set_write_image_lines(MIF<dtype> *self, int val)       {self->write_image_lines    = val;}
    void set_write_magnifications(MIF<dtype> *self, int val)    {self->write_magnifications = val;}
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
    dtype get_y1(MIF<dtype> *self)                              {return self->w0.re;}
    dtype get_y2(MIF<dtype> *self)                              {return self->w0.im;}
    dtype get_v1(MIF<dtype> *self)                              {return self->v.re;}
    dtype get_v2(MIF<dtype> *self)                              {return self->v.im;}
    int get_random_seed(MIF<dtype> *self)                       {return self->random_seed;}
    int get_write_stars(MIF<dtype> *self)                       {return self->write_stars;}
    int get_write_images(MIF<dtype> *self)                      {return self->write_images;}
    int get_write_image_lines(MIF<dtype> *self)                 {return self->write_image_lines;}
    int get_write_magnifications(MIF<dtype> *self)              {return self->write_magnifications;}
    const char* get_outfile_prefix(MIF<dtype> *self)            {return (self->outfile_prefix).c_str();}

    int get_num_stars(MIF<dtype> *self)                         {return self->get_num_stars();}
    dtype get_corner_x1(MIF<dtype> *self)                       {return self->get_corner().re;}
    dtype get_corner_x2(MIF<dtype> *self)                       {return self->get_corner().im;}
    dtype* get_stars(MIF<dtype> *self)                          {return &(self->get_stars()[0].position.re);}
    dtype* get_images(MIF<dtype> *self)                         {return &(self->get_images()[0].re);}
    int get_num_images(MIF<dtype> *self)                        {return self->get_num_images();}
    dtype* get_images_mags(MIF<dtype> *self)                    {return &(self->get_images_mags()[0].re);}
    dtype* get_image_lines(MIF<dtype> *self)                    {return &(self->get_image_lines()[0].re);}
    int get_num_image_lines(MIF<dtype> *self)                   {return self->get_num_image_lines();}
    int* get_image_lines_lengths(MIF<dtype> *self)              {return self->get_image_lines_lengths();}
    int get_total_image_lines_length(MIF<dtype> *self)          {return self->get_total_image_lines_length();}
    dtype* get_source_lines(MIF<dtype> *self)                   {return &(self->get_source_lines()[0].re);}
    dtype* get_image_lines_mags(MIF<dtype> *self)               {return self->get_image_lines_mags();}

    bool run(MIF<dtype> *self, int verbose)                     {return self->run(verbose);}
    bool save(MIF<dtype> *self, int verbose)                    {return self->save(verbose);}

    void MIF_delete(MIF<dtype> *self)                           {delete self;}

}