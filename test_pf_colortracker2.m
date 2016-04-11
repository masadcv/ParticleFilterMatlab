%% Demo illustrating Particle Filter applied to object tracking in a video
% 
% Likelihood is based on the mixture of 
% i) the Battacharya distance between a reference color pdf and current
% particule color pdf
% ii) gradient variation through normal vectors to an ellipsoid
%
% 
% 
%%  State Definition
%
%    S_k = A_k S_{k-1} + N(0 , R_k)
%
%    S_k = (x_k , x'_k , y_k , y'_k , H_k^x , H_k^y , theta_k)
%
%
%          |1   delta_k       0        0       0      0     0|
%          |0         1       0        0       0      0     0|
%          |0         0       1  delta_k       0      0     0|
% A  =     |0         0       0        1       0      0     0|
%          |0         0       0        0       1      0     0|
%          |0         0       0        0       0      1     0|
%          |0         0       0        0       0      0     1|
%
% R_k  =   |R_y       0   |
%          |0         R_e |
%
%                 |delta_k^3/3   delta_k^2/2       0        0 |
% R_y  = sigma_y  |delta_k^2/2       delta_k       0        0 |, % Kinematic of
%                 |0         0       delta_k^3/3   delta_k^2/2|  % the ellipsoid
%                 |0         0       delta_k^2/2       delta_k|

%          |sigma_Hx^2        0          0|
% R_e  =   |0        sigma_Hy^2          0|, % Kinematic of
%          |0         0      sigma_theta^2|  % the ellipsoid

clear all, close all hidden

%video_file       = 'seventh.avi';
%video_file       = 'Second_Xvid.avi';
video_file        = 'Second.avi';

savegif           = 0;
video             = mmreader(video_file);
offset_frame      = 80;%80
nb_frame          = get(video, 'numberOfFrames') - offset_frame - 10;
%nb_frame          = 400;
dim_x             = get(video , 'Width');
dim_y             = get(video , 'Height');


verbose           = 1;          % Display frames during filtering
hsv               = 1;          % Convert to HSV Color space
background        = 0;          % Use background Color Cue
mixing            = 1;          % weigthing cue
hist_col          = 1;          % Optimize color histograms
N                 = 250;        % Number of particules
N_threshold       = 6.*N/10;    % Redistribution threshold
delta             = 0.7;        % delta time

%%%%% Color Cue parameters %%%%%%%%

Npdf              = 120;       % Number of samples to draw inside ellipse to evaluate color histogram
Nx                = 6;         % Number of bins in first color dimension (R or H)
Ny                = 6;         % Number of bins in second color dimension (G or S)
Nz                = 6;         % Number of bins in third color dimension (B or V)
sigma_color       = 0.20;      % Measurement Color noise
nb_hist           = 256;

%%%%% Shape Cue parameters %%%%%%%%

h0                = 0.2;       % Non Detection probability along one of the l lines of the N ellipses
lambda            = 3.0;       % Mean number of detection along line
sigma_shape       = 0.5;       % Noise mesurement standart deviation
l                 = 25;        % Number of line dividing each ellipses
alpha0            = 11;        % Number of point discretizing each line (odd)
ratio             = 0.65;      % Ratio value to determine the Threshold of the detected values among line
bary              = 0.40;      % Parameter determining extramum points of each segment lines
nb_classe         = 40;        % Number of bin to build the gradient shape histogram
threshold_grad    = 30;

%%%%% Mixing Cue parameters %%%%%%%%

a                 = 1.0;  %pixel^(-1) % More a is big, less we forgot %
epsilon           = 1;
if (hsv)
    range         = 1;
else
    range         = 255;
end

pos_index         = [1 , 3];
ellipse_index     = [5 , 6 , 7];
d                 = 7;
M                 = Nx*Ny*Nz;
vect_col          = (0:range/(nb_hist - 1):range);
edge1             = (0 : range/Nx : range);
edge2             = (0 : range/Ny : range);
edge3             = (0 : range/Nz : range);

%%%%%% Target Localization for computing the target distribution %%%%

% Second.avi %
yq                = [185 ; 100];
eq                = [14 ; 20 ; pi/3];
% Seven.avi %
% yq                = [160 ; 95];
% eq                = [15 ; 15 ;0];

%%%%%% Background Localization for computing the background distribution %%%%

% Second.avi %
yb                = [160 ; 200];
eb                = [14 ; 20 ; pi/3];
% Seven.avi %
% yb                = [100 ; 150];
% eb                = [15 ; 15 ; 0];

%%%%%% Initialization distribution initialization %%%%

Sk                = zeros(d , 1);
Sk(pos_index)     = yq;
Sk(ellipse_index) = eq;

% Initial State covariance %

sigmax1           = 60;      % pixel %
sigmavx1          = 1;       % pixel / frame %
sigmay1           = 60;      % pixel  %
sigmavy1          = 1;       % pixel / frame %
sigmaHx1          = 4;       % pixel %
sigmaHy1          = 4;       % pixel %
sigmatheta1       = 8*(pi/180); % rad/frame %

% State Covariance %

% a) Position covariance %

sigmay            = 0.35;

% b) ellipse covariance %

sigmaHx           = 0.1;                % pixel %
sigmaHy           = 0.1;                % pixel %
sigmatheta        = 3.0*(pi/180);       % rad/frame %

%%%%%%%%%%%%%%%%%%%% State transition matrix %%%%%%%%%%%%%%%%%%%%%%

A                 = [1 delta 0 0 0 0 0 ; 0 1 0 0 0 0 0 ; 0 0 1 delta 0 0 0; 0 0 0 1 0 0 0 ; 0 0 0 0 1 0 0 ; 0 0 0 0 0 1 0 ; 0 0 0 0 0 0 1];
By                = [1 0 0 0 0 0 0 ; 0 0 1 0 0 0 0];
Be                = [0 0 0 0 1 0 0 ; 0 0 0 0 0 1 1 ; 0 0 0 0 0 0 1 ];

%%%%%% Initial State Covariance %%%%%

R1                = diag([sigmax1 , sigmavx1 , sigmay1 , sigmavy1 , sigmaHx1 , sigmaHy1 , sigmatheta1].^2);

%%%%%% State Covariance %%%%%

Rk                = zeros(d , d);
Ry                = sigmay*[delta^3/3 delta^2/2 0 0 ; delta^2/2 delta 0 0 ; 0 0 delta^3/3 delta^2/2 ; 0 0 delta^2/2 delta];
Re                = [sigmaHx.^2 0 0 ; 0 sigmaHy.^2 0 ; 0 0 sigmatheta.^2];
Rk(1 : 4  , 1 : 4)= Ry;
Rk(5 : d , 5 : d) = Re;
Ck                = chol(Rk)';
H                 = halton(d , N);
HH                = halton(2 , Npdf);
qmcpol            = [HH(1 , :).*cos(2*pi*HH(2 , :)) ; HH(1 , :).*sin(2*pi*HH(2 , :))];
Bk                = sqrt(2)*erfinv(2*H - 1);
Wk                = Ck*Bk;

%%%%%%%% Memory Allocation %%%%%%%

ON                = ones(1 , N);
Od                = ones(d , 1);
Smean             = zeros(d , nb_frame);
Pcov              = zeros(d , d , nb_frame);
py                = zeros(M , N);
N_eff             = zeros(1 , nb_frame);
a1                = zeros(1 , nb_frame + 1);
q1                = zeros(1 , nb_frame + 1);
a2                = zeros(1 , nb_frame + 1);
q2                = zeros(1 , nb_frame + 1);
threshold         = zeros(1 , nb_frame);
cte               = 1/N;
cteN              = cte(1 , ON);
w                 = cteN;
compteur          = 0;
cte1_color        = 1/(2*sigma_color*sigma_color);
cte2_color        = (1/(sqrt(2*pi)*sigma_color));
cte_mixing        = (delta/epsilon);
a1(1)             = 0.5;
a2(1)             = 0.5;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Target Distribution %%%%%%%%%%%%%%

I                     = read(video , offset_frame + 1);
Z                     = double(I);
if (hsv)
    im                = rgb2hsv_mex(Z);
else
    im                = Z;
end

if (hist_col)
    C1                = cumsum(histc(reshape(im(: , : , 1) , dim_x*dim_y , 1) , vect_col))/(dim_x*dim_y);
    C2                = cumsum(histc(reshape(im(: , : , 2) , dim_x*dim_y , 1) , vect_col))/(dim_x*dim_y);
    C3                = cumsum(histc(reshape(im(: , : , 3) , dim_x*dim_y , 1) , vect_col))/(dim_x*dim_y);
    i1                = sum(C1(: , ones(1 , Nx)) < repmat((0:1/(Nx - 1) : 1) , nb_hist  , 1));
    i2                = sum(C2(: , ones(1 , Ny)) < repmat((0:1/(Ny - 1) : 1) , nb_hist  , 1));
    i3                = sum(C3(: , ones(1 , Nz)) < repmat((0:1/(Nz - 1) : 1) , nb_hist  , 1));
    edge1             = [0 , vect_col(i1(2 : end)) , range];
    edge2             = [0 , vect_col(i2(2 : end)) , range];
    edge3             = [0 , vect_col(i3(2 : end)) , range];
end

%q                     = pdfcolor_ellipserand(im , yq , eq , Npdf , edge1 ,edge2 , edge3);
q                     = pdfcolor_ellipseqmc(im , yq , eq , qmcpol , edge1 , edge2 , edge3);
Q                     = q(: , ON);

if (background)
    %b                 = pdfcolor_ellipserand(im , yb , eb , Npdf , edge1 , edge2 , edge3);
    b                 = pdfcolor_ellipseqmc(im , yb , eb , qmcpol , edge1 , edge2 , edge3);
    B                 = b(: , ON);
end

%ellipseq          = ndellipse(yq , eq);

if (verbose)
    
    fig1              = figure(1);
    image(I);
    set(gca , 'drawmode' , 'fast');
    set(gcf , 'doublebuffer','on');
    set(gcf , 'renderer' , 'zbuffer');   
    if(savegif)
        gif_add_frame(gca,'video.gif');
    end  
end


%%%%%%%%%%%% Particle initialisation %%%%%%%%

%Sk                = Sk(: , ON) + chol(R1)'*randn(d , N);
Sk                = Sk(: , ON) + chol(R1)'*Bk;

%%%%%%%%%%%% Main Loop %%%%%%%%%%%%%%%%%%%%%%

for k = 1 : nb_frame ;
    
    fprintf('Frames = %1.0f/%1.0f\n' , k , nb_frame );
    
    I                = read(video , offset_frame + k);
    Z                = double(I);
    if (hsv)
        im           = rgb2hsv_mex(Z);
    else
        im           =  Z;
    end
    
    % --  Sk               = A*Sk + Wk;  --%
    
    Sk               = A*Sk + Ck*randn(d , N);    
    yk               = Sk(pos_index , :);     %yk                = By*Sk;
    ek               = Sk(ellipse_index , :); %ek                = Be*Sk;
    
    %%%%%%%%%%  Color Likelihood %%%%%%%%%
    
    %    [py , zi , yi]   = pdfcolor_ellipserand(im , yk , ek , Npdf , edge1 , edge2 , edge3);
    [py , zi , yi]       = pdfcolor_ellipseqmc(im , yk , ek , qmcpol , edge1 , edge2 , edge3);
    if (background)
        rho_py_q         = sum(sqrt(py.*Q));
        rho_py_b         = sum(sqrt(py.*B));
        likelihood_color = cte2_color*exp((rho_py_q - rho_py_b)*cte1_color);
    else
        rho_py_q         = sum(sqrt(py.*Q));
        likelihood_color = cte2_color*exp((rho_py_q - 1)*cte1_color);
    end
    
    likelihood_color     = likelihood_color/sum(likelihood_color);
   
    %%%%%%%%%%  Gradient-Shape Likelihood %%%%%%%%%
    
    %    likelihood_shape = pdfgrad_ellipse(Z , yk , ek , h0 , lambda , sigma_shape , l , alpha0 , ratio , bary , nb_classe , threshold_grad);
    
    likelihood_shape     = pdfgrad_ellipse(Z , yk , ek , h0 , lambda , sigma_shape , l , alpha0 , ratio , bary , nb_classe);

    %%%%%%%%%%  Compute Mixing weigth cue  %%%%%%%%%
    
    prod_like            = likelihood_color.*likelihood_shape;
    
    if (mixing)
        tmp1             = (likelihood_color - sum(likelihood_color)*cte);
        tmp2             = (likelihood_shape - sum(likelihood_shape)*cte);
        tmp3             = (prod_like - sum(prod_like)*cte);
        tmp4             = sum(tmp3.*tmp3);
        coeff1           = abs(sum(tmp1.*tmp3)/sqrt(sum(tmp1.*tmp1)*tmp4));
        coeff2           = abs(sum(tmp2.*tmp3)/sqrt(sum(tmp2.*tmp2)*tmp4));
        q1(k + 1)        = exp(-a*(1 - coeff1));
        q2(k + 1)        = exp(-a*(1 - coeff2));
        sq               = 1/(q1(k + 1) + q2(k + 1));
        q1(k + 1)        = q1(k + 1)*sq;
        q2(k + 1)        = q2(k + 1)*sq;
        a1(k + 1)        = a1(k) + (q1(k + 1) - a1(k))*cte_mixing;
        a2(k + 1)        = a2(k) + (q2(k + 1) - a2(k))*cte_mixing;
        w                = w.*(likelihood_color.^a1(k + 1)).*(likelihood_shape.^a2(k + 1));
    else
        w                = w.*prod_like;
    end
    
    w                    = w/sum(w);
    
    %--------------------------- 6) MMSE estimate & covariance -------------------------
    
    [Smean(: , k) , Pcov(: , : , k)] = part_moment(Sk , w);
    
    %--------------------------- 7) Particles redistribution ? if N_eff < N_threshold -------------------------
    
    N_eff(k)                         = 1./sum(w.*w);
    
    if (N_eff(k) < N_threshold)
        compteur              = compteur + 1;
        indice_resampling     = particle_resampling(w);    
        % Copy particules
        Sk                    = Sk(: , indice_resampling);
        w                     = cteN;
    end
    
    %%%%%%%%%%%%% Display %%%%%%%%%%%%%%%
    
    if (verbose)
        fig1              = figure(1);
        image(I);
        title(sprintf('N = %6.3f/%6.3f, Frame = %d, Redistribution =%d' , N_eff(k) , N_threshold , k , compteur));
        ind_k             = (1 : k);
        hold on
        ykmean            = Smean(pos_index , k);
        ekmean            = Smean(ellipse_index, k);
        [xmean , ymean]   = ellipse(ykmean , ekmean);
        
        %plot(ellipsemean(1 , :) , ellipsemean(2 , :) , 'r' , ellipseq(1 , :) , ellipseq(2 , :) , 'linewidth' , 3)
        
        plot(xmean , ymean , 'g' , 'linewidth' , 3)
        plot(Smean(pos_index(1) , ind_k) , Smean(pos_index(2) , ind_k) , 'r' , 'linewidth' , 2)
        
        %         qui = quiver(Smean(pos_index(1) , k) , Smean(pos_index(2) , k) , Smean(ellipse_index(2) , k)*sin(-Smean(ellipse_index(3) , k)) , Smean(ellipse_index(2) , k)*cos(Smean(ellipse_index(3) , k)) , 'c');
        %         set(qui , 'linewidth' , 2);
        
        %plot(reshape(yi(1 , 1 , :) , 1 , N) , reshape(yi(2 , 1 , :) , 1 , N), '+')
        
        plot(Sk(pos_index(1) , :) , Sk(pos_index(2) , :) , 'b+');
        
        %imshow(edge(rgb2gray(im) , 'canny') + rgb2gray(im))
        
        hold off
        if(savegif)
            gif_add_frame(gca , 'video.gif');
        end
        
        %         fig2 = figure(2);
        %         plot3(reshape(yi(1 , 1 , :) , 1 , N) , reshape(yi(2 , 1 , :) , 1 , N), rho_py_q , '+')
        %         xlabel('x (m)');
        %         ylabel('y (m)');
        %         zlabel('\rho(p_{y},q)');
        %         axis([1 , dim_x , 1 , dim_y , 0 , 1]);
        %         axis ij
        %         grid on
        %         fig3 = figure(3)
        %         plot(1:k,a1(1:k),1:k,a2(1:k) , 'r')
        %
        
    end
    
    % [Xi , Yi] = meshgrid((1:dim_y) , (1:dim_x));
    % Zi        = griddata(squeeze(yi(1 , 1 , :)) , squeeze(yi(2 , 1 , :)) , rho_py_q , Xi , Yi , 'cubic');
    % surfc(Xi , Yi , Zi);
    % shading interp
    %  trisurf(delaunay(squeeze(yi(1 , 1 , :)) , squeeze(yi(2 , 1 , :))) , squeeze(yi(1 , 1 , :)) , squeeze(yi(2 , 1 , :)) , rho_py_q)
    % shading interp
    % axis([1 , dim_x , 1 , dim_y , 0 , 1]);
    
    drawnow;
end

%% Display
figure(3)
plot(Smean(pos_index(1) , :) , Smean(pos_index(2) , :) , 'linewidth' , 2)
axis([1 , dim_x , 1 , dim_y ]);
axis ij
grid on
title('Helicopter trajectory');

figure(4)
plot((1 : nb_frame) , Smean(ellipse_index(end) , :) , 'linewidth' , 2)
axis([1 , nb_frame , -pi , pi ]);
xlabel('Frames k');
ylabel('\theta');
grid on
title('Ellipse angle versus frames');

figure(5);
h = slice(reshape(q , Nx , Ny , Nz) , (1 : Nx) , (1 : Ny) , (1 : Nz));
colormap(flipud(cool));
alpha(h , 0.1);
brighten(+0.5);
title('3D Histogram of the Target distribution');
xlabel('Bin x');
ylabel('Bin y');
zlabel('Bin z');
colorbar
cameramenu;

if (hist_col)
    figure(6);
    hold on
    plot(vect_col , C1 , vect_col , C2, vect_col , C3);
    plot(edge1 , C1([1 , i1(2 : end) , nb_hist]) , '+' , edge2 , C2([1 , i2(2 : end) , nb_hist])   ,'*' , edge3 , C3([1 , i3(2 : end) , nb_hist]) ,'p')
    hold off
    ylabel('HSV CDF');
end
if (mixing)
    figure(7)
    plot((1:nb_frame) , a1(2:end) , (1:nb_frame) , a2(2:end) , 'r');
    axis([1 , nb_frame , 0 , 1]);
    grid on
    legend('Color Cue' , 'Gradient Cue' , 0);
    title(sprintf('a = %2.1f, \\epsilon = %2.1f' , a , epsilon))
end