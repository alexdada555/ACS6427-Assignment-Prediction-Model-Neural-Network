clear all
close all
clc;

%% Load Data

load concrete.mat

%%  Split data in to Training Data and OOS Data

n = 1030/2;
[x,z,x_star,z_star] = dmrndsplit(D(:,1:8),D(:,end),n); % Randomly divide into 50% Training and 50% OOS

% Colums 1 - 8 = x
% Cement; Blast Furnace Slag; Fly Ash; Water; Super-plasticizer; Coarse Aggregate; Fine Aggregate; Age in days;
% Column 9 = z
% CCS (MPa)in the ninth column
%% normalization

[x,m,s] = dmstandard(x);          % Normalize Training Data
[x_star]= dmstandard(x_star,m,s); % Normalize OOS Data

%%  Benchmark
% In this instance a general linear model is used as a benchmark to train
% on data which will be compared against the subsequent models trained on the 
% same data set.

myglm = glm(size(x,2),1,'linear');
options = foptions;
[myglm] = glmtrain(myglm,options,x,z); % train glm
y_star_hat = glmfwd(myglm,x_star);     % evaluate oos

PI = sum((y_star_hat-z_star).^2)/sum((z_star-mean(z_star)).^2); %Compute PI (normalised sse)

figure(1);
dmscat(z_star,y_star_hat)    % plot corrolation between OOS output and model output
title("Data-Model Correlation")
legend("Data","Model")
figure(2);
dmplotres(z_star,y_star_hat) % plot of errors
title("Data-Model Errors")
legend("Error")
figure(3);
histfit(z_star-y_star_hat)   % plot of error hystogrm showing spread across range
title("Data-Model Error Histogram")

%%  First Data model: MLP

nhid = 20; % MLP Hidden Layer Elements
k = 3;    % k for k-fold Cross Validation
rho = logspace(-1,1,50); % rho Grid 
nits = 100; % Iterations to Allow for Minimum Reach

starttime = tic;

% Perform Cross Validated Training of the MLP

PI = zeros(length(rho),1);  % Create vector to store peformace index

for l = 1:length(rho)%For Each Value of rho in the Distribution....
	options = foptions; % Initialize Options
	options(1) = 0;     % Set to Minimal Output
	options(14) = nits; % Set Epoch to nits Value
	mymlp = mlp(size(x,2),nhid,1,'linear',rho(l)); % Initialize MLP
	y_hat = dmxval(mymlp,options,x,z,k);           % Calculate CV Model Output
	PI(l) = sum((y_hat-z).^2)/sum((z-mean(z)).^2); % Calculate Model Perfomance
end

disp('finished loop')
disp([num2str(toc(starttime)/60,2) ' min elapsed'])

figure(4);
semilogx(rho,PI);
xlabel('ln (\rho)');
ylabel('nsse')

idx = find(PI == min(PI));  % Find Index of Lowest Error 
rho_min = rho(idx);         % value of Lowest CV Error

% Retrain model using rho that provided minimum error

options = foptions; % initialize options
options(1) = 0;     % set "silent"
options(14) = nits; % ensure enough iterations allowed
mymlp = mlp(size(x,2),nhid,1,'linear',rho_min); % initialize mymlp
mymlp = mlptrain(mymlp,options,x,z);% train mlp
y_hat = mlpfwd(mymlp,x);            % evaluate on TRAINING sample
y_star_hat = mlpfwd(mymlp,x_star);  % evaluate oos

%% compute PI (normalised sse) for all TRAINING data
% as a check against any "local minimum" problems

PI_chk = sum((y_hat-z).^2)/sum((z-mean(z)).^2);
disp(['cross validated nsse = ' num2str(PI(idx),3)])

disp(['      retrained nsse = ' num2str(PI_chk,3)])

figure(5);
dmscat(z_star,y_star_hat)    % plot corrolation between OOS output and model output
title("Data-Model Correlation")
legend("Data","Model")
figure(6);
dmplotres(z_star,y_star_hat) % plot of errors
title("Data-Model Errors")
legend("Error")
figure(7);
histfit(z_star-y_star_hat)   % plot of error hystogrm showing spread across range
title("Data-Model Error Histogram")

%% Second Model Using GRBF
% The performance of the MLP is going to be compared against that of the
% GRBF

nhid = 10; % choose # of hidden units

options = foptions; % initialize options
options(1) = 0; % set "verbose"
options(14) = 10; % set # EM iterations

myrbf = rbf(size(x,2),nhid,1,'gaussian'); % initialize myrbf
[myrbf,options] = rbftrain(myrbf,options,x,z); % train rbf
y_star_hat = dmxval(myrbf,options,x,z,k); % evaluate oos
y_hat = mlpfwd(mymlp,x);            % evaluate on TRAINING sample
%y_star_hat = mlpfwd(mymlp,x_star);  % evaluate oos
PI = sum((y_star_hat-z).^2)/sum((z-mean(z)).^2);

% compare observed with predicted

figure(8);
dmscat(z_star,y_star_hat)    % plot corrolation between OOS output and model output
title("Data-Model Correlation")
legend("Data","Model")
figure(9);
dmplotres(z_star,y_star_hat) % plot of errors
title("Data-Model Errors")
legend("Error")
figure(10);
histfit(z_star-y_star_hat)   % plot of error hystogrm showing spread across range
title("Data-Model Error Histogram")

%%  Save Data

save('dmmi160140802.mat','myglm','mymlp','myrbf','x_star','z_star');

