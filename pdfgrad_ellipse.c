
/*
 Compute the likelihood of the gradient variations through normal vectors to an ellipsoid. 
  

 Usage 
 -----

  [likelihood , zi , yi , yp , threshold]   = pdfgrad_ellipse(Z , y , e , h0 , lambda , sigma , l , alpha , ratio , bary , nb_bins);


  Inputs
  ------

   Z               Image (m x n x 3) in double format
   y               Center position of the ellipsoid (2 x N) where designs the number of particles
   e               Ellipsoid (3 x N) where designs the number of particles
   h0              Constante in likelihood computation (default h0 = 0.2)
   lambda          Constante in likelihood computation (default lambda = 0.2)
   sigma           standard deviation of the RBF in the likelihood model (default sigma = 0.7)
   l               Number of points to discretize gradient normal vector to the ellipse (default l = 25)
   alpha           Number of points to discretize the ellipse (default alpha = 11)
   ratio           Quantile to estimate gradient's threshold (default ratio = 0.9)
   bary            Barycenter (default bary = 0.2)
   nb_bins         Number of bins to compute gradient threshold (default nb_bins = 32)

  Ouputs
  -------

  likelihood       Likelihood 
  zi               Interpolated color values (3 x Npf x N)
  py               pdf color (NxNyNz x N)
  yp               Position of Interpolated values (2 x Npf x N)
  threshold        Estimated threshold



 To compile
 -----------

 mex pdfgrad_ellipse.c

 mex -f mexopts_intel10.bat -output pdfgrad_ellipse.dll pdfgrad_ellipse.c


Example 1
---------

mov         = aviread('Second.avi' , 1);
Z           = mov.cdata;
nR          = 320;
nC          = 240;
N           = 5000;
%Z           = double(ceil(255*rand(nR , nC , 3)));
y           = repmat([nR ; nC] , 1 , N) + randn(2 , N);
e           = rand(3 , N);
%y           = [111 , 52 , 15 , 56 ; 22 , 100 , 34 , 43];
%e           = [10 , 10 , 10 , 30 ; 15, 3 , 22 , 10 ; -pi/1.1 , pi/2 , 0 ,0.75*pi];
%N           = 4;
h0          = 0.1;
lambda      = 2;
sigma       = 1;
l           = 20;
alpha       = 11;
ratio       = 0.8;
bary        = 0.1;
nb_bins   = 32;

[likelihood , zi , yi , yp]   = pdfgrad_ellipse(Z , y , e , h0 , lambda , sigma , l , alpha , ratio , bary , nb_bins);%
set(gcf ,'renderer' , 'zbuffer');
plot3(reshape(yi(1 , : , : , :) , alpha*l*N , 1) , reshape(yi(2 , : , : , :) , alpha*l*N , 1) , reshape(zi(1 , : , : , :) , alpha*l*N , 1) , '+') 


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 2004

*/

#include <math.h>
#include "mex.h"


#define NUMERICS_FLOAT_MIN 1.0E-37
#define NUMERICS_FLOAT_MAX 1.0E+38
#define cteR 2.989360212937755e-001
#define cteG 5.870430744511213e-001
#define cteB 1.140209042551033e-001
#define PI 3.14159265358979323846
#define NOBIN -1

#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif


/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
/*------------------------ Headers   --------------------------------------*/
/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/

int bin(double , double * , int);

/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/

void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
	double *Z , *y , *e  ;
	double *likelihood , *zi , *yi , *yp;
	double *normgradzi , *edge ,  *thres;
	double *cos_angle , *sin_angle;
	double h0 , lambda , sigma  , ratio , bary;
	double twoPI = 2*PI , u , offset_bary , pas_bary , pas_edge , threshold;
	double offset_l , pas_l , zx , zy , posx , posy ;
	double angle , Hx , Hy, theta , cos_theta , sin_theta , tmp1 , tmp2 , s , t , dist2; 
	double cte1_sigma , cte2_sigma, cte_h0 , like;
	double ztmp1 , ztmp2 , ztmp3 , ztmp4; 
	double val , mini = NUMERICS_FLOAT_MAX , maxi = NUMERICS_FLOAT_MIN;
	const int *dimsZ, *dimsy , *dimse;
	int *dimslikelihood , *dimszi , *dimsyi , *dimsyp;
	int *pnormgrad;
	unsigned int l , alpha , nb_bins , cupnorm;
	unsigned int N , i , j , k  , n , m , nm;
	unsigned int kalphal , kalpha2 , kalpha3, jalpha, alphaj , alpha11 , k2 , k3 , i2 , j2;
	unsigned int numdimsZ , numdimsy , numdimse , ndx , kzi;
	unsigned int numdimsyi , numdimszi , numdimslikelihood , numdimsyp; 
	unsigned int floors , floort , compteur = 0 , thresh_compteur , thresh_notgiven;
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
		mexErrMsgTxt("Z must be at least a Matrix (n x m x 3)");
	}
	n         = dimsZ[0];
	m         = dimsZ[1];
	nm        = n*m;

	/*----------- Input 2 ---------------*/

	y         = mxGetPr(prhs[1]);
	numdimsy  = mxGetNumberOfDimensions(prhs[1]);
	dimsy     = mxGetDimensions(prhs[1]);
	if ( (numdimsy != 2) || (dimsy[0] != 2))
	{
		mexErrMsgTxt("y must be (2 x N)");
	}
	N        = dimsy[1];

	/*----------- Input 3 ---------------*/

	e        = mxGetPr(prhs[2]);
	numdimse = mxGetNumberOfDimensions(prhs[2]);
	dimse    = mxGetDimensions(prhs[2]);
	if ( (numdimse != 2) || (dimse[0] != 3) || (dimse[1] != N))
	{
		mexErrMsgTxt("e must be (3 x N)");
	}

	/*----------- Input 4 ---------------*/
	if(nrhs >= 4)
	{
		h0 = mxGetScalar(prhs[3]);	
	}
	else
	{
		h0 = 0.2;
	}
	/*----------- Input 5 ---------------*/

	if(nrhs >= 5)
	{	
		lambda = mxGetScalar(prhs[4]);	
	}
	else
	{
		lambda = 0.2;
	}

	/*----------- Input 6 ---------------*/

	if(nrhs >= 6)
	{	
		sigma = mxGetScalar(prhs[5]);	
	}
	else
	{
		sigma = 0.7;
	}

	/*----------- Input 7 ---------------*/

	if(nrhs >= 7)
	{	
		l = (int) mxGetScalar(prhs[6]);	
	}
	else
	{
		l = 25;
	}

	/*----------- Input 8 ---------------*/

	if(nrhs >= 8)
	{	
		alpha = (int) mxGetScalar(prhs[7]);	
	}
	else
	{
		alpha = 11;
	}

	/*----------- Input 9 ---------------*/

	if(nrhs >= 9)
	{	
		ratio = mxGetScalar(prhs[8]);	
	}
	else
	{
		ratio = 0.9;
	}

	/*----------- Input 10 ---------------*/

	if(nrhs >= 9)
	{	
		bary = mxGetScalar(prhs[9]);	
	}
	else
	{
		bary = 0.2;
	}

	/*----------- Input 11 ---------------*/

	if(nrhs >= 11)
	{	
		nb_bins = (int) mxGetScalar(prhs[10]);	
	}
	else
	{
		nb_bins = 32;
	}

	if(nrhs >= 12)
	{	
		threshold        = (int) mxGetScalar(prhs[11]);
		thresh_notgiven  = 0;
	}
	else
	{
		thresh_notgiven  = 1;
	}

	/*----------- Output 1 ---------------*/


	numdimslikelihood      = 2;
	dimslikelihood         = (int *)mxMalloc(numdimslikelihood*sizeof(int));
	dimslikelihood[0]      = 1;
	dimslikelihood[1]      = N;

	plhs[0]                = mxCreateNumericArray(numdimslikelihood, dimslikelihood, mxDOUBLE_CLASS, mxREAL);
	likelihood             = mxGetPr(plhs[0]);

	/*----------- Output 2 ---------------*/

	numdimszi              = 4;
	dimszi                 = (int *)mxMalloc(numdimszi*sizeof(int));
	dimszi[0]              = 1;
	dimszi[1]              = alpha;
	dimszi[2]              = l;
	dimszi[3]              = N;

	plhs[1]                = mxCreateNumericArray(numdimszi, dimszi, mxDOUBLE_CLASS, mxREAL);
	zi                     = mxGetPr(plhs[1]);

	/*----------- Output 3 ---------------*/

	numdimsyi              = 4;
	dimsyi                 = (int *)mxMalloc(numdimsyi*sizeof(int));
	dimsyi[0]              = 2;
	dimsyi[1]              = alpha;
	dimsyi[2]              = l;
	dimsyi[3]              = N;

	plhs[2]                = mxCreateNumericArray(numdimsyi, dimsyi, mxDOUBLE_CLASS, mxREAL);
	yi                     = mxGetPr(plhs[2]);

	/*----------- Output 4 ---------------*/

	numdimsyp              = 3;
	dimsyp                 = (int *)mxMalloc(numdimsyp*sizeof(int));
	dimsyp[0]              = 2;
	dimsyp[1]              = l;
	dimsyp[2]              = N;
	plhs[3]                = mxCreateNumericArray(numdimsyp, dimsyp, mxDOUBLE_CLASS, mxREAL);
	yp                     = mxGetPr(plhs[3]);

	/*----------- Output 5 ---------------*/

	plhs[4]                = mxCreateDoubleMatrix(1 ,1 , mxREAL);
	thres                  = mxGetPr(plhs[4]);

    /*----------- Temporaly matrices ---------------*/


	normgradzi             = (double *)malloc((alpha*l*N)*sizeof(double));       /* (1 x alpha x l x N) */
	edge                   = (double *)malloc((nb_bins + 1)*sizeof(double));   /* (1 x alpha + 1) */
	pnormgrad              = (int *)malloc((nb_bins)*sizeof(int));             /* (1 x alpha) */
	cos_angle              = (double *)malloc(l*sizeof(double));
	sin_angle              = (double *)malloc(l*sizeof(double));


	offset_bary            = (1.0 - bary);
	offset_l               = 0.0;
	pas_l                  = twoPI/(l - 1);
	pas_bary               = 2.0*bary/(alpha - 1);
	cte1_sigma             = 1.0/(2.0*sigma*sigma);
	cte2_sigma             = 1.0/(sqrt(twoPI)*sigma);
	cte_h0                 = cte2_sigma/(sqrt(twoPI)*h0*lambda);

	for(j = 0 ; j < l ; j++)
	{
		angle                = j*pas_l;
		cos_angle[j]         = cos(angle);
		sin_angle[j]         = sin(angle);
	}

	for (k = 0 ; k < N ; k++)
	{
		k2        = k*2;
		k3        = k2 + k;
		kalpha2   = k*alpha*l;
		kalphal   = 2*kalpha2;
		kalpha3   = k2*l;

		posx      = y[0 + k2];  /*(2 x N) */
		posy      = y[1 + k2];

		Hx        = e[0 + k3];
		Hy        = e[1 + k3];
		theta     = e[2 + k3];  /*(3 x N) */

		cos_theta = cos(theta);
		sin_theta = sin(theta);

		for (j = 0 ; j < l ; j++)
		{			
			j2                   = j*2;
			jalpha               = j2*alpha + kalphal;
			alphaj               = j*alpha;

			tmp1                 = Hx*cos_angle[j];
			tmp2                 = Hy*sin_angle[j];

			zx                   = posx + (cos_theta*tmp1 - sin_theta*tmp2);
			yp[0 + j2 + kalpha3] = zx;                /* (2 x l x N) */ 

			zy                   = posy + (sin_theta*tmp1 + cos_theta*tmp2); 
			yp[1 + j2 + kalpha3] = zy; 

			for (i = 0 ; i < alpha ; i++)
			{							
				i2                       = i*2;
				u                        = offset_bary + i*pas_bary;
				s                        = (1.0 - u)*posx + u*zx;
				t                        = (1.0 - u)*posy + u*zy;
				yi[0 + i2 + jalpha ]     = s;
				yi[1 + i2 + jalpha ]     = t;

				if ( (s < 1) || (s > m) || (t < 1) || (t > n) )
				{	
					zi[i + alphaj + kalpha2]  = NUMERICS_FLOAT_MIN;	
				}
				else
				{					
					floors                    = floor(s);
					floort                    = floor(t);
					ndx                       = floort + (floors - 1)*n - 1;
					s                         = (s - floors);
					t                         = (t - floort);
					ztmp1                     = Z[ndx]*cteR         + Z[ndx + nm]*cteG         + Z[ndx + 2*nm]*cteB;
					ztmp2                     = Z[ndx + 1]*cteR     + Z[ndx + nm + 1]*cteG     + Z[ndx + 2*nm + 1]*cteB;
					ztmp3                     = Z[ndx + n]*cteR     + Z[ndx + nm + n]*cteG     + Z[ndx + 2*nm + n]*cteB;
					ztmp4                     = Z[ndx + n + 1]*cteR + Z[ndx + nm + n + 1]*cteG + Z[ndx + 2*nm + n + 1]*cteB;
					val                       = (ztmp1*(1 - t) + ztmp2*t)*(1 - s) + ( ztmp3*(1 - t) + ztmp4*t )*s;
					zi[i + alphaj + kalpha2]  = val;  /* (1 x alpha x l x N) */
				}
			}
		}
	}

	/*  Gradient Evaluation */

	for (i = 0 ; i < l*N ; i++)
	{
		alpha11 = i*alpha;
		for (j = 0 ; j < (alpha - 1) ; j++)
		{		
			if ( (zi[j + 1 + alpha11] != NUMERICS_FLOAT_MIN) && ( zi[j + alpha11] != NUMERICS_FLOAT_MIN))
			{
				val                     = fabs(zi[j + 1 + alpha11] - zi[j + alpha11]); /* (1 x alpha x l x N) */
				normgradzi[j + alpha11] = val;
				mini                    = min(val , mini);
				maxi                    = max(val , maxi);	
			}
			else
			{
				normgradzi[j + alpha11] = NUMERICS_FLOAT_MIN;	
			}
		}
		if ( (zi[0 + alpha11] != NUMERICS_FLOAT_MIN) && (zi[j  + alpha11] != NUMERICS_FLOAT_MIN) )
		{
			val                     = fabs(zi[0 + alpha11] - zi[j  + alpha11]);
			normgradzi[j + alpha11] = val;
			mini                    = min(val , mini);
			maxi                    = max(val , maxi);
		}
		else 
		{
			normgradzi[j + alpha11] = NUMERICS_FLOAT_MIN;
		}
	}

	/* Threshold estimation */

	if (thresh_notgiven)
	{
		pas_edge     = (maxi - mini)/(nb_bins);
		for (i = 0 ; i < (nb_bins + 1) ; i++)
		{
			edge[i] = mini + i*pas_edge;	
		}
		for (i = 0 ; i < nb_bins ; i++)
		{
			pnormgrad[i]      =  0;
		}
		for(i = 0 ; i < alpha*l*N ;  i ++)
		{			
			if (normgradzi[i] != NUMERICS_FLOAT_MIN)
			{
				kzi                  =  bin(normgradzi[i] , edge , nb_bins);
				pnormgrad[kzi]++;
				compteur++;
			}
		}
		cupnorm         = 0;
		i               = 0;
		thresh_compteur = floor(compteur*ratio);
		while( (cupnorm < thresh_compteur) && (i < nb_bins + 1) ) /* (alpha*N*l*ratio) */
		{
			cupnorm += pnormgrad[i];
			i++;	
		}
		threshold = edge[i];
	}
	val   = 0.0;
	for (k = 0 ; k < N ; k++)
	{				
		k2             = k*2;
		kalphal        = k*alpha*l;
		kalpha2        = k2*alpha*l;
		kalpha3        = k2*l;
		alpha11        = k*l;
		likelihood[k]  = 1.0;

		for (j = 0 ; j < l ; j++)
		{			
			j2        = j*2;
			jalpha    = j2*alpha  + kalpha2;
			alphaj    = j*alpha   + kalphal;
			posx      = yp[0 + j2 + kalpha3];   /*  (2 x l x N);   */  
			posy      = yp[1 + j2 + kalpha3];
			like      = 0.0;
			for (i = 0 ; i < alpha ; i++)
			{
				i2    = i*2;
				if((normgradzi[i + alphaj] > threshold) && (normgradzi[i + alphaj] != NUMERICS_FLOAT_MIN)) /* (1 x alpha x l x N) */
				{
					k2    = i2 + jalpha;
					tmp1  = (yi[0 + k2 ] - posx);       /* (2 x alpha x l x N) */
					tmp2  = (yi[1 + k2 ] - posy);
					dist2 = (tmp1*tmp1 + tmp2*tmp2);
					like += exp(-dist2*cte1_sigma );				
				}		
			}
			likelihood[k]           *= (1.0 + cte_h0*like);
		}	
		val           += likelihood[k];
	}

	val               = 1.0/val;
	for(k = 0 ; k < N ; k++)
	{
		likelihood[k] *= val;
	}
	thres[0]            = threshold;

	free(normgradzi);
	free(edge);
	free(pnormgrad);
	free(cos_angle);
	free(sin_angle);
}
/*-------------------------------------------------------------------------------------------------*/
int bin(double zi , double *vect_edge , int Nbin)
{
/*
vect_edge (1 x Nbin + 1)
*/
	int k  = NOBIN, k0 = 0 , k1 = Nbin;

	if ((zi >= vect_edge[0]) && (zi <= vect_edge[k1]))
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
/*-------------------------------------------------------------------------------------------------*/
