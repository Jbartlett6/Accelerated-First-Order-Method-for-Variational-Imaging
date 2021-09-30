function [u,v] = tsvl2_optimizer_easy(im1,im2,maxIter,lambda,tol,acc)

A = double(imread('yos9.tif'));
B = double(imread('yos10.tif'));
%A = double(imread('card1.bmp'));
%B = double(imread('card2.bmp'));
A = imresize(A, 0.5);
B = imresize(B, 0.5);
A = A(:,:,1)/255;
B = B(:,:,1)/255;
im1 = A;
im2 = B;

tol = 1e-10;
acc = 0;
maxIter = 500;


lambda = 0.003;
theta_1 = 1;   % convergnece parameter
theta_2 = 0.01; % convergnece parameter
alpha = 1.8; % relaxation parameter
beta = 3e3*lambda;
gamma = 1;



%Initialising the kernels and derivatives of the images:
[Ix, Iy] = computeDerivatives(im1);
It = im1 - im2;


It = im1-im2;
J11 = Ix.*Ix;
J22 = Iy.*Iy;
J12 = Ix.*Iy;
J21 = Iy.*Ix;
J13 = Ix.*It;
J23 = Iy.*It;

%Initialising the variables at zero:
[m,n]=size(im1);
%b is the dual variable:
b11=zeros(m,n); b12=zeros(m,n);
b21=zeros(m,n); b22=zeros(m,n);

d1 =zeros(m,n); d2 =zeros(m,n);
w11=zeros(m,n); w12=zeros(m,n);
w21=zeros(m,n); w22=zeros(m,n);
p11=zeros(m,n); p12=zeros(m,n);
p21=zeros(m,n); p22=zeros(m,n);
Wx =zeros(m,n); Wy =zeros(m,n);
u =zeros(m,n);  v =zeros(m,n);

%Creating the matricies for the fft inversion - requires change to apply to
%TSV/TGV.
[Y,X]=meshgrid(0:n-1,0:m-1);
G1=cos(pi*Y/n)-1;
G2=cos(pi*X/m)-1;
a11 = 1 + theta_1-2*beta*(G1);
a22 = 1 + theta_1-2*beta*(G2);

for i=1 : maxIter
    
    %temp_u = u;
    %temp_v = v;   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [u, v] = resolvent_operator(Wx-d1, Wy-d2, Ix, Iy, It, J11, J22, theta_2);
    %u = ((J22 + theta_2).*(Wx-d1) - J12.*(Wy-d2) - J13) ./ (J11 + J22 + theta_2);
    %v = ((J11 + theta_2).*(Wy-d2) - J21.*(Wx-d1) - J23) ./ (J11 + J22 + theta_2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %if acc == 1
        %u = alpha * u + (1-alpha) * Wx;
        %v = alpha * v + (1-alpha) * Wy;
    %end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update u1 using dct
    div_w_b=Bx(w11-b11+p11)+By(w12-b12+p12);
    g=theta_2*(u+d1)-theta_1*div_w_b;
    Wx=mirt_idctn(mirt_dctn(g)./(theta_2-2*theta_1*(G1+G2)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %update p using fourier inverse:
    p11=mirt_idctn((mirt_dctn(theta_1*(Fx(Wx)+b11-w11)))./a11);
    p12=mirt_idctn((mirt_dctn(theta_1*(Fy(Wx)+b12-w12)))./a22);

    % update u2 using dct
    div_w_b=Bx(w21-b21+p21)+By(w22-b22+p22);
    g=theta_2*(v+d2)-theta_1*div_w_b;
    Wy=mirt_idctn(mirt_dctn(g)./(theta_2-2*theta_1*(G1+G2)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %update p using the fourier inverse
    p21=mirt_idctn((mirt_dctn(theta_1*(Fx(Wy)+b21-w21)))./a11);
    p22=mirt_idctn((mirt_dctn(theta_1*(Fy(Wy)+b22-w22)))./a22);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    %Soft Thresholding, to update w:
    c11=Fx(Wx)-p11+b11;
    c12=Fy(Wx)-p12+b12;
    c21=Fx(Wy)-p21+b21;
    c22=Fy(Wy)-p22+b22;
    %abs_c=sqrt(c11.^2+c12.^2+c21.^2+c22.^2+eps);
    %w11=max(abs_c-lambda/theta_1,0).*c11./abs_c;
    %w12=max(abs_c-lambda/theta_1,0).*c12./abs_c;
    %w21=max(abs_c-lambda/theta_1,0).*c21./abs_c;
    %w22=max(abs_c-lambda/theta_1,0).*c22./abs_c;
    w11=max(abs(c11)-lambda/theta_1,0).*sign(c11);
    w12=max(abs(c12)-lambda/theta_1,0).*sign(c12);
    w21=max(abs(c21)-lambda/theta_1,0).*sign(c21);
    w22=max(abs(c22)-lambda/theta_1,0).*sign(c22);
         
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update Bregman iterative parameters b
    b11=c11-w11;
    b12=c12-w12;
    b21=c21-w21;
    b22=c22-w22;
    d1=d1+u-Wx;
    d2=d2+v-Wy;
    
    EV(i) = Primal_Eval_TSV(u, v, p11, p12, p21, p22, Ix, Iy, It, lambda, beta, gamma);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % convergnece check
    %stop1 = sum(sum(abs(u-temp_u)))/(sum(sum(abs(temp_u)))+eps);
    %stop2 = sum(sum(abs(v-temp_v)))/(sum(sum(abs(temp_v)))+eps);
    %stop(i) = max(stop1, stop2);
    %if stop(i) < tol
        %if i > 2
            %fprintf('    iterate %d times, stop due to converge to tolerance \n', i);
            %break; % set break crmaxIterion
        %end
    %end
end
%visualization
figure; imagesc(A); colormap(gray); title('DVF'); axis off; axis equal;
hold on;
opflow = opticalFlow(u,v);
plot(opflow, 'DecimationFactor',[3 3],'ScaleFactor',10);
q = findobj(gca,'type','Quiver');
q.Color = 'r';
q.LineWidth = 1;

figure; imagesc(flowToColor(u,v)); title('HSV'); axis off; axis equal;


function Fxu = Fx(u)
[m,n] = size(u);
Fxu = circshift(u,[0 -1])-u;
Fxu(:,n) = zeros(m,1);

% Forward derivative operator on y with boundary condition u(1,:,:)=u(m,:,:)
function Fyu = Fy(u)
[m,n] = size(u);
Fyu = circshift(u,[-1 0])-u;
Fyu(m,:) = zeros(1,n);

% Backward derivative operator on x with boundary condition Bxu(:,1)=u(:,1)
function Bxu = Bx(u)
[~,n] = size(u);
Bxu = u - circshift(u,[0 1]);
Bxu(:,1) = u(:,1);
Bxu(:,n) = -u(:,n-1);

% Backward derivative operator on y with boundary condition Bxu(1,:)=u(1,:)
function Byu = By(u)
[m,~] = size(u);
Byu = u - circshift(u,[1 0]);
Byu(1,:) = u(1,:);
Byu(m,:) = -u(m-1,:);

function [ux, uy]=computeDerivatives(u)
[m,n]=size(u);
C1 = circshift(u,[0 -1]); C1(:,n) = C1(:,n-1);
C2 = circshift(u,[0 1]);  C2(:,1) = C2(:,2);
C3 = circshift(u,[-1 0]); C3(m,:) = C3(m-1,:);
C4 = circshift(u,[1 0]);  C4(1,:) = C4(2,:);
ux=(C1-C2)/2;
uy=(C3-C4)/2;

function [u, v] = resolvent_operator(tmp1, tmp2, Ix, Iy, It, J11, J22, theta_2)

J11 = Ix.*Ix;
J22 = Iy.*Iy;
J12 = Ix.*Iy;
J21 = Iy.*Ix;
J13 = Ix.*It;
J23 = Iy.*It;

 u = ((J22 + theta_2).*(tmp1) - J12.*(tmp2) - J13) ./ (J11 + J22 + theta_2);
 v = ((J11 + theta_2).*(tmp2) - J21.*(tmp1) - J23) ./ (J11 + J22 + theta_2);

%((1/theta_2)eye(n*m)-)*-theta_2*(tmp1)-It

%rho_w = Ix .* tmp1 + Iy .* tmp2 + It;
%zhat = theta_2 * rho_w ./ (J11 + J22 + eps);
%tmp = zhat./max(abs(zhat), 1);
%u = tmp1 - (Ix / theta_2).*tmp;
%v = tmp2 - (Iy / theta_2).*tmp;


function EV = Primal_Eval_TSV(u, v, p11, p12, p21, p22, Ix, Iy, It, alpha, beta, gamma)
    %Anisotropic    
    %A = 0.5*(Ix.*u + Iy.*v + It).^2                     ...
    %+ alpha * (abs(Fx(u) - p11) + abs(Fy(u) - p12))     ...
    %+ alpha * (abs(Fx(v) - p21) + abs(Fy(v) - p22))     ...
    %+ 0.5*beta * (Fx(p11).^2 + Fy(p12).^2)              ...
    %+ 0.5*beta * (Fx(p21).^2 + Fy(p22).^2)              ...
    %+ 0.5*gamma*(p11.^2 + p12.^2)                       ...
    %+ 0.5*gamma*(p21.^2 + p22.^2);
    
    %Isotropic
    A = 0.5*(Ix.*u + Iy.*v + It).^2                     ...
    + alpha * ((Fx(u) - p11).^2 + (Fy(u) - p12).^2      ...
    + (Fx(v) - p21).^2 + (Fy(v) - p22).^2)              ...
    + 0.5*beta * (Fx(p11).^2 + Fy(p12).^2)              ...
    + 0.5*beta * (Fx(p21).^2 + Fy(p22).^2)              ...
    + 0.5*gamma*(p11.^2 + p12.^2)                       ...
    + 0.5*gamma*(p21.^2 + p22.^2);
    


    EV = sum(sum(A));