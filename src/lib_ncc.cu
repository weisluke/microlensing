#include "ncc.cuh"


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
    
    LIB_EXPORT NCC<dtype>* NCC_init()                                      {return new NCC<dtype>();}

    LIB_EXPORT void set_infile_prefix(NCC<dtype> *self, const char* val)   {self->infile_prefix        = val;}
    LIB_EXPORT void set_center_y1(NCC<dtype> *self, dtype val)             {self->center_y.re          = val;}
    LIB_EXPORT void set_center_y2(NCC<dtype> *self, dtype val)             {self->center_y.im          = val;}
    LIB_EXPORT void set_half_length_y1(NCC<dtype> *self, dtype val)        {self->half_length_y.re     = val;}
    LIB_EXPORT void set_half_length_y2(NCC<dtype> *self, dtype val)        {self->half_length_y.im     = val;}
    LIB_EXPORT void set_num_pixels_y1(NCC<dtype> *self, int val)           {self->num_pixels_y.re      = val;}
    LIB_EXPORT void set_num_pixels_y2(NCC<dtype> *self, int val)           {self->num_pixels_y.im      = val;}
    LIB_EXPORT void set_over_sample(NCC<dtype> *self, int val)             {self->over_sample          = val;}
    LIB_EXPORT void set_write_maps(NCC<dtype> *self, int val)              {self->write_maps           = val;}
    LIB_EXPORT void set_write_histograms(NCC<dtype> *self, int val)        {self->write_histograms     = val;}
    LIB_EXPORT void set_outfile_prefix(NCC<dtype> *self, const char* val)  {self->outfile_prefix       = val;}

    LIB_EXPORT const char* get_infile_prefix(NCC<dtype> *self)             {return (self->infile_prefix).c_str();}
    LIB_EXPORT dtype get_center_y1(NCC<dtype> *self)                       {return self->center_y.re;}
    LIB_EXPORT dtype get_center_y2(NCC<dtype> *self)                       {return self->center_y.im;}
    LIB_EXPORT dtype get_half_length_y1(NCC<dtype> *self)                  {return self->half_length_y.re;}
    LIB_EXPORT dtype get_half_length_y2(NCC<dtype> *self)                  {return self->half_length_y.im;}
    LIB_EXPORT int get_num_pixels_y1(NCC<dtype> *self)                     {return self->num_pixels_y.re;}
    LIB_EXPORT int get_num_pixels_y2(NCC<dtype> *self)                     {return self->num_pixels_y.im;}
    LIB_EXPORT int get_over_sample(NCC<dtype> *self)                       {return self->over_sample;}
    LIB_EXPORT int get_write_maps(NCC<dtype> *self)                        {return self->write_maps;}
    LIB_EXPORT int get_write_histograms(NCC<dtype> *self)                  {return self->write_histograms;}
    LIB_EXPORT const char* get_outfile_prefix(NCC<dtype> *self)            {return (self->outfile_prefix).c_str();}

    LIB_EXPORT int* get_num_crossings(NCC<dtype> *self)                    {return self->get_num_crossings();}
    
    LIB_EXPORT bool run(NCC<dtype> *self, int verbose)                     {return self->run(verbose);}
    LIB_EXPORT bool save(NCC<dtype> *self, int verbose)                    {return self->save(verbose);}

    LIB_EXPORT void NCC_delete(NCC<dtype> *self)                           {delete self;}

}