/*
  Compute the color pdf of an image inside an ellipsoid of parameters y and e
  pdf is computed by quasi monte-carlo approximation via halton sequence

  Usage 
  -----

  [py , zi , yi]   = pdfcolor_ellipseqmc(Z , y , e , qmcpol , [vect_edge1] , [vect_edge2] , [vect_edge3] );%


  Inputs
  ------

   Z               Image (m x n x 3) in double format
   y               Center position of the ellipsoid (2 x N) where designs the number of particles
   e               Ellipsoid radius and angle (3 x N) where designs the number of particles
   qmcpol          1 x Npdf (default Npdf = 100)
   vect_edge1      Color support (R or H) (1 x Nx) (default vect_edge1 = (0:1/8:1))
   vect_edge2      Color support (G or S) (1 x Ny) (default vect_edge2 = (0:1/8:1))
   vect_edge3      Color support (B or V) (1 x Nz) (default vect_edge2 = (0:1/8:1))

  Ouputs
  -------

  py               Color pdf (NxNyNz x N)
  zi               Interpolated color values (3 x Npf x N)
  yi               Position of Interpolated values (2 x Npf x N)

  To compile
  -----------

  mex pdfcolor_ellipseqmc.c 

  mex -f mexopts_intel10.bat -output pdfcolor_ellipseqmc.dll pdfcolor_ellipseqmc.c

  Example 1
  ---------
	
  Z                = rand(200 , 200 , 3);
  y                = [111 , 52 , 15 , 56 ; 22 , 100 , 34 , 43];
  e                = [10 , 10 , 10 , 30 ; 15, 3 , 22 , 10 ; -pi/1.1 , pi/2 , 0 ,0.75*pi];
  Npdf             = 4000;
  Nx               = 4;
  Ny               = 3;
  Nz               = 5;
  M                = Nx*Ny*Nz;
  vect_edge1       = (0 : 1/Nx : 1);
  vect_edge2       = (0 : 1/Ny : 1);
  vect_edge3       = (0 : 1/Nz : 1);
  h                = halton1(2 , Npdf);
  qmcpol           = [h(1 , :).*cos(2*pi*h(2 , :)) ; h(1 , :).*sin(2*pi*h(2 , :))];
  [py , zi , yi]   = pdfcolor_ellipseqmc(Z , y , e , qmcpol , vect_edge1 , vect_edge2 , vect_edge3 );%
  figure(1)
  imagesc(Z);
  hold on
  plot(squeeze(yi(1 , : , :)) , squeeze(yi(2 , : , :)) , '+')
  axis ij
  axis equal
  hold off
  figure(2)
  plot((1:M) , py);
  axis([1 , M , 0 , 0.12])
	
 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 2004
	  
*/

#include <math.h>
#include "mex.h"

#define NUMERICS_FLOAT_MIN 1.0E-37
#define PI 3.14159265358979323846
#define NOBIN -1

#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif


/*------------------------ Headers   --------------------------------------*/

int bin(double , double * , int);

/*-------------------------------------------------------------------------*/

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	double *Z , *y , *e , *qmcpol ;
	double *py , *zi , *yi;
	double *vect_edge1 , *vect_edge2 , *vect_edge3;
	double cte1 , cte2 , cte3 , val , posx , posy ;
	double Hx , Hy , theta , cos_theta , sin_theta , tmp1 , tmp2 , s , t , invM = 1.0 , NotANumber = mxGetNaN();
	const int *dimsZ , *dimsy , *dimse , *dimsqmcpol , *dimsvect_edge1 , *dimsvect_edge2 , *dimsvect_edge3;
	int *dimspy=NULL , *dimszi=NULL , *dimsyi=NULL;
	int i , j , n , m , nm , r , N , Npdf , NNpdf , Nx , Ny , Nz , Nxy;
	int numdimsZ , numdimsy , numdimse , numdimsqmcpol, numdimsvect_edge1 , numdimsvect_edge2 , numdimsvect_edge3;
	int numdimspy , numdimszi , numdimsyi; 
	int iNpdf2, iNpdf3 , iM , i2 , i3 , j2 , j3 , ndx , kx , ky , kz , M , j2iNpdf2 , j3iNpdf3;
	int floors , floort , compteur;

	/*---------------------------------------------------------------*/
	/*---------------------- PARSE INPUT ----------------------------*/	
	/*---------------------------------------------------------------*/

	if(nrhs < 3)
	{
		mexErrMsgTxt("At least 3 inputs are required");		
	}

	/*----------- Input 1 ---------------*/

	Z        = mxGetPr(prhs[0]);
	numdimsZ = mxGetNumberOfDimensions(prhs[0]);
	dimsZ    = mxGetDimensions(prhs[0]);
	if (numdimsZ != 3)
	{
		mexErrMsgTxt("Z must be a matrix (n x m x 3)");
	}
	n         = dimsZ[0];
	m         = dimsZ[1];
	nm        = n*m;

	/*----------- Input 2 ---------------*/

	y         = mxGetPr(prhs[1]);
	numdimsy  = mxGetNumberOfDimensions(prhs[1]);
	dimsy     = mxGetDimensions(prhs[1]);
	if ((numdimsy != 2) || (dimsy[0] != 2))
	{	
		mexErrMsgTxt("y input be (2 x N)");
	}
	N        = dimsy[1];

	/*----------- Input 3 ---------------*/

	e        = mxGetPr(prhs[2]);
	numdimse = mxGetNumberOfDimensions(prhs[2]);
	dimse    = mxGetDimensions(prhs[2]);
	if ( (numdimse != 2) || (dimse[0] != 3) || (dimse[1] != N) )
	{	
		mexErrMsgTxt("e must be (3 x N)");
	}

	/*----------- Input 4 ---------------*/

	qmcpol           = mxGetPr(prhs[3]);
	numdimsqmcpol    = mxGetNumberOfDimensions(prhs[3]);
	dimsqmcpol       = mxGetDimensions(prhs[3]);
	if ((numdimsqmcpol != 2) || (dimsqmcpol[0] != 2))
	{	
		mexErrMsgTxt("qmcpol must be (2 x Npdf)");
	}	
	Npdf  = dimsqmcpol[1];
	NNpdf = Npdf*N;

	/*----------- Input 5 ---------------*/
	if(nrhs >= 5)
	{
		vect_edge1        = mxGetPr(prhs[4]);
		numdimsvect_edge1 = mxGetNumberOfDimensions(prhs[4]);
		dimsvect_edge1    = mxGetDimensions(prhs[4]);
		if ((numdimsvect_edge1 != 2) || ((dimsvect_edge1[0] != 1) && (dimsvect_edge1[1] > 1)) || ((dimsvect_edge1[1] != 1) && (dimsvect_edge1[0] > 1)))
		{	
			mexErrMsgTxt("vect_edge1 must be (1 x (Nx + 1)) or ((Nx + 1)x1)");	
		}
		Nx    = max(dimsvect_edge1[0] , dimsvect_edge1[1]) - 1;
	}
	else
	{
		vect_edge1 = (double *)malloc(1*9*sizeof(double));
		Nx         = 8;
		cte1       = 1.0/Nx;
		for (i = 0; i < Nx + 1 ; i++)
		{		
			vect_edge1[i] = i*cte1;
		}
	}

	/*----------- Input 6 ---------------*/

	if(nrhs >= 6)
	{
		vect_edge2        = mxGetPr(prhs[5]);
		numdimsvect_edge2 = mxGetNumberOfDimensions(prhs[5]);
		dimsvect_edge2    = mxGetDimensions(prhs[5]);
		if ((numdimsvect_edge2 != 2) || ((dimsvect_edge2[0] != 1) && (dimsvect_edge2[1] > 1)) || ((dimsvect_edge2[1] != 1) && (dimsvect_edge2[0] > 1)))
		{	
			mexErrMsgTxt("vect_edge2 must be (1 x (Ny + 1)) or ((Ny + 1)x1)");	
		}
		Ny    = max(dimsvect_edge2[0] , dimsvect_edge2[1]) - 1;
	}
	else
	{
		vect_edge2 = (double *)mxMalloc(1*9*sizeof(double));
		Ny         = 8;
		cte2       = 1.0/Ny;
		for (i = 0 ; i < Ny + 1 ; i++)
		{		
			vect_edge2[i] = i*cte2;
		}
	}

	/*----------- Input 7 ---------------*/

	if(nrhs >= 7)
	{
		vect_edge3        = mxGetPr(prhs[6]);
		numdimsvect_edge3 = mxGetNumberOfDimensions(prhs[6]);
		dimsvect_edge3    = mxGetDimensions(prhs[6]);
		if ((numdimsvect_edge3 != 2) || ((dimsvect_edge3[0] != 1) && (dimsvect_edge3[1] > 1)) || ((dimsvect_edge3[1] != 1) && (dimsvect_edge3[0] > 1)))
		{	
			mexErrMsgTxt("vect_edge3 must be (1 x (Nz + 1)) or ((Nz + 1)x1)");	
		}
		Nz    = max(dimsvect_edge3[0] , dimsvect_edge3[1]) - 1;
	}
	else
	{
		vect_edge3 = (double *)malloc(1*9*sizeof(double));
		Nz         = 8;
		cte3       = 1.0/Nz;
		for (i = 0 ; i < Nz + 1 ; i++)
		{		
			vect_edge3[i] = i*cte3;
		}
	}
	Nxy            = Nx*Ny;
	M              = Nxy*Nz;
	invM           = 1.0/(double)(M);

	/*----------- Output 1 ---------------*/

	numdimspy      = 2;
	dimspy         = (int *)mxMalloc(numdimspy*sizeof(int));
	dimspy[0]      = M;
	dimspy[1]      = N;
	plhs[0]        = mxCreateNumericArray(numdimspy, dimspy, mxDOUBLE_CLASS, mxREAL);
	py             = mxGetPr(plhs[0]);

	/*----------- Output 2 ---------------*/

	numdimszi      = 3;
	dimszi         = (int *)mxMalloc(numdimszi*sizeof(int));
	dimszi[0]      = 3;
	dimszi[1]      = Npdf;
	dimszi[2]      = N;
	plhs[1]        = mxCreateNumericArray(numdimszi, dimszi, mxDOUBLE_CLASS, mxREAL);
	zi             = mxGetPr(plhs[1]);

	/*----------- Output 3 ---------------*/

	numdimsyi      = 3;
	dimsyi         = (int *)mxMalloc(numdimsyi*sizeof(int));
	dimsyi[0]      = 2;
	dimsyi[1]      = Npdf;
	dimsyi[2]      = N;
	plhs[2]        = mxCreateNumericArray(numdimsyi, dimsyi, mxDOUBLE_CLASS, mxREAL);
	yi             = mxGetPr(plhs[2]);

	/* --------- Compute ellipsoïds Coordinates & Interpolate data & retrieved Histograms---------------*/

	for (i = 0 ; i < N ; i++)
	{
		i2         = i*2;
		i3         = i2 + i;
		iM         = i*M;
		iNpdf2     = i2*Npdf;
		iNpdf3     = i3*Npdf;
		posx       = y[0 + i2];  /*(2 x N) */
		posy       = y[1 + i2];
		Hx         = e[0 + i3];
		Hy         = e[1 + i3];
		theta      = e[2 + i3];
		cos_theta  = cos(theta);
		sin_theta  = sin(theta);
		compteur   = 0;
		for (j = 0 ; j < Npdf ; j++)
		{
			j2                  = j + j;
			j2iNpdf2            = j2 + iNpdf2;
			j3                  = j2 + j;
			j3iNpdf3            = j3 + iNpdf3;
			tmp1                = Hx*qmcpol[0 + j2];
			tmp2                = Hy*qmcpol[1 + j2];
			s                   = posx + (cos_theta*tmp1 - sin_theta*tmp2);
			t                   = posy + (sin_theta*tmp1 + cos_theta*tmp2);
			yi[0 + j2iNpdf2]    = s;
			yi[1 + j2iNpdf2]    = t;
			if ((s < 1) || (s > m) || (t < 1) || (t > n))
			{
				zi[0 + j3iNpdf3]              = NotANumber;
				zi[1 + j3iNpdf3]              = NotANumber;
				zi[2 + j3iNpdf3]              = NotANumber;
			}
			else
			{
				floors                        = floor(s);
				floort                        = floor(t);
				ndx                           = floort + (floors - 1)*n - 1;
				s                             = (s - floors);
				t                             = (t - floort);
				r                             = 0;
				val                           = (Z[ndx + r]*(1 - t) + Z[ndx + r + 1]*t)*(1 - s) + ( Z[ndx + r + n]*(1 - t) + Z[ndx + n + 1 + r]*t )*s;
				zi[0 + j3iNpdf3]              = val;
				kx                            = bin(val , vect_edge1 , Nx);

				r                             = nm;
				val                           = (Z[ndx + r]*(1 - t) + Z[ndx + r + 1]*t)*(1 - s) + ( Z[ndx + r + n]*(1 - t) + Z[ndx + n + 1 + r]*t )*s;
				zi[1 + j3iNpdf3]              = val;
				ky                            = bin(val , vect_edge2 , Ny);

				r                             = nm + nm;
				val                           = (Z[ndx + r]*(1 - t) + Z[ndx + r + 1]*t)*(1 - s) + ( Z[ndx + r + n]*(1 - t) + Z[ndx + n + 1 + r]*t )*s;
				zi[2 + j3iNpdf3]              = val;
				kz                            = bin(val , vect_edge3 , Nz);

				py[kx + ky*Nx + kz*Nxy + iM]++;
				compteur++;
			}
		}
		if (compteur != 0)
		{
			invM        = 1.0/compteur;
			for (j = 0 ; j < M ; j++)
			{
				py[j + iM] *= invM;
			}			
		}
	}
	if(nrhs < 7)
	{
		free(vect_edge3);
	}
	if(nrhs < 6)
	{
		free(vect_edge2);
	}
	if(nrhs < 5)
	{
		free(vect_edge1);
	}

	mxFree(dimspy);
	mxFree(dimszi);
	mxFree(dimsyi);
}
/*------------------------------------------------------------*/
int bin(double zi , double *vect_edge , int Nbin)
{
	int k  = NOBIN, k0 = 0 , k1 = Nbin ;
	if ( (zi >= vect_edge[0]) && (zi <= vect_edge[k1]) )
	{
		k = (k0 + k1)/2;
		while (k0 < k1 - 1)
		{
			if (zi >= vect_edge[k]) 
			{
				k0 = k;
			}
			else 
			{
				k1 = k;
			}
			k = (k0 + k1)/2;
		}
		k = k0;
	}
	return k;  
}
/*------------------------------------------------------------*/




