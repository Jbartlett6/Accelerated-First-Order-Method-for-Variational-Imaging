function [u,v] = Dual_prox_alg()
    clear all
    close

    %Loading and normalising the image:
    source = double(imread('lenaNoise.bmp'));
    %source = double(imread('smooth1.bmp'));
    f0 = source(:,:,1) / 255;
    [m, n] = size(f0);

    %Setting the smoothing parameter and calculating the step size:
    lambda = 0.2;
    step_size = 1/8*lambda^2;

    %Intitialising the variables for the algorithm:
    prev_p = zeros(m,n,2);
    curr_p = zeros(m,n,2);
    velocity = zeros(m,n,2);
    prev_theta = 1;
    v1 = 0;

    for i = 1:5000
        %Updating theta
        curr_theta = (1+sqrt(1+4*prev_theta^2))/2;

        %Step forward according to momentum and velocity
        p = prev_p + (prev_theta-1)/curr_theta * velocity;

        %Perform the gradient update on p:   
        div =  Bx(p(:,:,1))+By(p(:,:,2)); 
        grad_p = (1/lambda)*(cat(3,Fx(f0),Fy(f0))) + (cat(3,Fx(div),Fy(div)));
        curr_p = p + step_size*grad_p;

        %Project back into the unit ball (proximal step):
        eu_sq = sqrt(curr_p(:,:,1).^2+curr_p(:,:,2).^2);
        curr_p = cat(3,curr_p(:,:,1)./max(eu_sq,1),curr_p(:,:,2)./max(eu_sq,1));
        velocity = curr_p - prev_p;

        %u(i+1) = sum(abs(curr_p(:)-prev_p(:)))/sum(abs(curr_p(:)));
        %score(i) = Evalu(f0 + lambda*(Bx(curr_p(:,:,1))+By(curr_p(:,:,2))));

        %Restart decision:
        if sum((p(:)-curr_p(:)).*v1(:))>0
            curr_theta = 1;
        end
        
        %Calculating v1 to be used for the next restart decision:
        v1 = curr_p-prev_p;

        %Updating variables for the next iteration:
        prev_p = curr_p;
        prev_theta = curr_theta;

        %Recording the duality gap for this iteration:
        u11 = f0 + lambda*(Bx(curr_p(:,:,1))+By(curr_p(:,:,2)));
        u(i) = Evalu(u11)-Dual_Eval(curr_p);

    end

    u11 = f0 + lambda*(Bx(curr_p(:,:,1))+By(curr_p(:,:,2)));
    figure; imagesc(u11); colormap(gray); axis off; axis equal;
    figure; plot(log(u));



function output = prox_grad(p, f0, lambda)
    p1 = p(:,:,1);
    p2 = p(:,:,2);
    div_p = div(p1, p2);
    q1 = p1 + Fx(div_p+f0/lambda)/8;
    q2 = p2 + Fy(div_p+f0/lambda)/8;
    q = max(sqrt(q1.^2 + q2.^2),1);
    output = cat(3, q1./q, q2./q);

%Forward x derivative
function Fxu = Fx(u)
    [m,n] = size(u);
    Fxu = circshift(u,[0 -1])-u;
    Fxu(:,n) = zeros(m,1);

%Forward y derivative
function Fyu = Fy(u)
    [m,n] = size(u);
    Fyu = circshift(u,[-1 0])-u;
    Fyu(m,:) = zeros(1,n);

%Backward x derivative
function Bxu = Bx(u)
    [~,n] = size(u);
    Bxu = u - circshift(u,[0 1]);
    Bxu(:,1) = u(:,1);
    Bxu(:,n) = -u(:,n-1);

%Backward y derivative
function Byu = By(u)
    [m,~] = size(u);
    Byu = u - circshift(u,[1 0]);
    Byu(1,:) = u(1,:);
    Byu(m,:) = -u(m-1,:);

%Div operator
function f = div(a,b)
    f = Bx(a) + By(b);

%Evaluating the primal objective function
function Ev = Evalu(u)
    source = double(imread('lenaNoise.bmp'));
    lambda = 0.2;
    f0 = source(:,:,1) / 255;
    dx = Fx(u);
    dy = Fy(u);
    gradu_norm = sqrt((dx).^2+(dy).^2);
    n = sum((u-f0).^2,'all');
    Ev = 0.5*n+lambda*sum(gradu_norm(:));

%Evaluating the Dual Objective Function
function duev = Dual_Eval(u)
    source = double(imread('lenaNoise.bmp'));
    lambda = 0.2;
    f0 = source(:,:,1) / 255;

    div = Bx(u(:,:,1))+By(u(:,:,2));
    first = -lambda*sum(div.*f0,'all');

    div_norm = sum(div.*div,'all');

    second = -((lambda^2)/2)*div_norm;
    duev = first + second;


        