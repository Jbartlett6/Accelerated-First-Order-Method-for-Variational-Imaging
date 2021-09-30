function energy = TSVL2_OF_PrimalDual()
close all
A = double(imread('yos9.tif'));
B = double(imread('yos10.tif'));
%A = double(imread('card1.bmp'));
%B = double(imread('card2.bmp'));
A = imresize(A,0.5);
B = imresize(B,0.5);

A = A(:,:,1)/255;
B = B(:,:,1)/255;

figure; imagesc(A); colormap(gray); title('original source image'); axis off; axis equal;
figure; imagesc(B); colormap(gray); title('original target image'); axis off; axis equal;

[m,n] = size(A);


%Calculating the derivatives which will be used in the data term:
[Ix, Iy] = computeDerivatives(A);
It = A - B;
J11 = Ix.*Ix;
J22 = Iy.*Iy;

%initialisint hyperparameters
alpha = 0.003;
beta = alpha*2e2;
gamma = 1;

%Setting step size
u_step_size = 1/(8*alpha^2);
p_step_size = (1/(4*beta + gamma+alpha));
q_step_size = 0.01;

%Initialising Variables
prev_u = zeros(m,n);
prev_v = zeros(m,n);
prev_p11 = zeros(m,n); prev_p12 = zeros(m,n);
prev_p21 = zeros(m,n); prev_p22 = zeros(m,n);
prev_q11 = zeros(m,n); prev_q12 = zeros(m,n);
prev_q21 = zeros(m,n); prev_q22 = zeros(m,n);


for i = 1:1000
    %u update
    u_tilde = prev_u + u_step_size*(alpha*div(prev_q11, prev_q12));
    v_tilde = prev_v + u_step_size*(alpha*div(prev_q21, prev_q22));
    [curr_u, curr_v] = prox_u(u_tilde, v_tilde, Ix, Iy, It, J11, J22, 1/u_step_size);
    
    %q1 update
    curr_p11 = prev_p11 - p_step_size*(-alpha*prev_q11 - beta*Bx(Fx(prev_p11))+gamma*prev_p11);
    curr_p12 = prev_p12 - p_step_size*(-alpha*prev_q12 - beta*By(Fy(prev_p12))+gamma*prev_p12);
    %q2 update
    curr_p21 = prev_p21 - p_step_size*(-alpha*prev_q21 - beta*Bx(Fx(prev_p21))+gamma*prev_p21);
    curr_p22 = prev_p22 - p_step_size*(-alpha*prev_q22 - beta*By(Fy(prev_p22))+gamma*prev_p22);
    
    %Interpolation step
    u_ba = curr_u; %+ (curr_u - prev_u);
    v_ba = curr_v;
    p11_ba = curr_p11; %+ (curr_p1 - prev_p1);
    p12_ba = curr_p12; %+ (curr_p2 - prev_p2);
    p21_ba = curr_p21;
    p22_ba = curr_p22;
    
    %q1 gradient update:
    q11_tilde = prev_q11 + q_step_size*(Fx(u_ba) - p11_ba);
    q12_tilde = prev_q12 + q_step_size*(Fy(u_ba) - p12_ba);
    %q2 gradient update:
    q21_tilde = prev_q21 + q_step_size*(Fx(v_ba) - p21_ba);
    q22_tilde = prev_q22 + q_step_size*(Fy(v_ba) - p22_ba);
    
    %proximal step for q:
    [curr_q11, curr_q12, curr_q21, curr_q22] = proj(q11_tilde, q12_tilde, q21_tilde, q22_tilde);
    
    
    
    %Evaluating the objective energy value
    energy(i) = Primal_Eval_TSV(curr_u, curr_v, curr_p11, curr_p12, curr_p21, curr_p22, Ix, Iy, It, alpha, beta, gamma);
    
    %Updating the variables fro the next iteration
    prev_u = curr_u; prev_v = curr_v;
    prev_p11 = curr_p11; prev_p12 = curr_p12;
    prev_p21 = curr_p21; prev_p22 = curr_p22;
    prev_q11 = curr_q11; prev_q12 = curr_q12;
    prev_q21 = curr_q21; prev_q22 = curr_q22;
end    
%visualization
figure; imagesc(A); colormap(gray); title('DVF'); axis off; axis equal;
hold on;
opflow = opticalFlow(curr_u,curr_v);
plot(opflow, 'DecimationFactor',[3 3],'ScaleFactor',10);
q = findobj(gca,'type','Quiver');
q.Color = 'r';
q.LineWidth = 1;

figure; imagesc(flowToColor(curr_u,curr_v)); title('HSV'); axis off; axis equal;
figure; plot(log(energy));
energy(i)
    
    
function [ux, uy]=computeDerivatives(u)
[m,n]=size(u);
C1 = circshift(u,[0 -1]); C1(:,n) = C1(:,n-1);
C2 = circshift(u,[0 1]);  C2(:,1) = C2(:,2);
C3 = circshift(u,[-1 0]); C3(m,:) = C3(m-1,:);
C4 = circshift(u,[1 0]);  C4(1,:) = C4(2,:);
ux=(C1-C2)/2;
uy=(C3-C4)/2;

%The partial difference equations
% Forward derivative operator on x with boundary condition u(:,:,1)=u(:,:,1)
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

%Divergence Function
function d = div(a,b)
    d = Bx(a) + By(b);

  
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
    + alpha * sqrt(((Fx(u) - p11).^2 + (Fy(u) - p12).^2      ...
    + (Fx(v) - p21).^2 + (Fy(v) - p22).^2))              ...
    + 0.5*beta * (Fx(p11).^2 + Fy(p12).^2)              ...
    + 0.5*beta * (Fx(p21).^2 + Fy(p22).^2)              ...
    + 0.5*gamma*(p11.^2 + p12.^2)                       ...
    + 0.5*gamma*(p21.^2 + p22.^2);
    


    EV = sum(sum(A));
     
function [u, v] = prox_u(tmp1, tmp2, Ix, Iy, It, J11, J22, theta_2)
J11 = Ix.*Ix;
J22 = Iy.*Iy;
J12 = Ix.*Iy;
J21 = Iy.*Ix;
J13 = Ix.*It;
J23 = Iy.*It;

 u = ((J22 + theta_2).*(tmp1) - J12.*(tmp2) - J13) ./ (J11 + J22 + theta_2);
 v = ((J11 + theta_2).*(tmp2) - J21.*(tmp1) - J23) ./ (J11 + J22 + theta_2);
 
    
    function [q11, q12, q21, q22] = proj(a, b, c, d)
%Anisotropic
%q11 = a./max(abs(a),1);
%q12 = b./max(abs(b),1);
%q21 = c./max(abs(c),1);
%q22 = d./max(abs(d),1);
norm = sqrt(a.^2 + b.^2 +c.^2 +d.^2);
q11 = a./max(norm,1);
q12 = b./max(norm,1);
q21 = c./max(norm,1);
q22 = d./max(norm,1);

%p = cat(3,p1,p2);