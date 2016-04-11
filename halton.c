/* 

  Generate quasi-montecarlo Halton sequence

  
  Usage          x = halton(d , N);
  ------    


  Inputs         d   dimension (d < 11)
  ------
                 N   number of points to generate


  Ouput          x   halton sequence (d x N)
  -----
  
	
  compile with: mex -f mexopts_intel10.bat -output halton.dll halton.c
  -----------

  test in matlab :
  --------------

  x = halton(2 , 500);
  plot(x(1 , :) , x(2 , :) , '+')


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/02/2006

 Reference ""


*/

#include "mex.h"
#include "math.h"


#include "mex.h"
#include "math.h"

/*-------------------------------------------------------------------------------------------------*/

void halton(int , int , double *);

/*-------------------------------------------------------------------------------------------------*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,  const mxArray *prhs[])
{
	double *h;
	int  dim , nbpts;

	if((nrhs < 1) ||  (nrhs > 2)  )    
	{
		mexErrMsgTxt("Usage: h = halton1(dim , nbpts);");
	}
	/* --- Input 1 ---*/

	dim = (int) mxGetScalar(prhs[0]);
	if(dim > 11 )    
	{
		mexErrMsgTxt("halton1(dim , nbpts) with dim = 1,...,11");
	}

	/* --- Input 2 ---*/

	nbpts = (int) mxGetScalar(prhs[1]);

	/* --- Output 1 ---*/

	plhs[0]    = mxCreateDoubleMatrix(dim , nbpts ,  mxREAL);
	h          = mxGetPr(plhs[0]);

	/*---------- Main Call --------- */

	halton(dim , nbpts , h );

	/*------------------------------ */
}
/*-------------------------------------------------------------------------------------------------*/
void halton(int dim , int nbpts, double *h)
{
	double lognbpts , d , sum;
	int i , j , n , t , b;
	static int P[11] = {2 ,3 ,5 , 7 , 11 , 13 , 17 , 19 ,  23 , 29 , 31};
	double *p;

	p          = (double *)malloc((ceil(log(nbpts + 1)/log(2)))*sizeof(double));

	lognbpts   = log(nbpts + 1);
	for(i = 0 ; i < dim ; i++)
	{
		b      = P[i];
		n      = (int) ceil(lognbpts/log(b));
		for(t = 0 ; t < n ; t++)
		{
			p[t] = pow(b , -(t + 1) );
		}
		for (j = 0 ; j < nbpts ; j++)
		{
			d        = j + 1;
			sum      = fmod(d , b)*p[0];
			for (t = 1 ; t < n ; t++)
			{
				d        = floor(d/b);
				sum     += (fmod(d , b)*p[t]); 
			}
			h[i + j*dim] = sum;
		}
	}
	free(p);
}
/*-------------------------------------------------------------------------------------------------*/
