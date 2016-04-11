try
    echo on
    mex ellipse.c 
    mex rgb2hsv_mex.c
    mex pdfcolor_ellipserand.c
    mex pdfcolor_ellipseqmc.c
    mex pdfgrad_ellipse.c
    mex halton.c
    mex part_moment.c
    mex particle_resampling.c
    echo off
catch exception
    if(~isempty(exception))
        fprintf(['\n Error during compilation, be sure to:\n'...
            'i)  You have C compiler installed (prefered compiler are MSVC/Intel/GCC)\n'...
            'ii) You did "mex -setup" in matlab prompt before running mexme_pf_color_tracker\n']);
    end
end

