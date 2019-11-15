% MATLAB program to initialize a parallel pool,
% used for interaction with the MATLAB engine in python.
% Defines global variables in a MATLAB session.

%%
% Initialize workspace
clear
clc
close all hidden


%%
% Assert no already started cluster
delete(gcp('nocreate'));

% Start local cluster
local_cluster = parcluster();

% Choose number of workers
number_of_workers = 32;
local_cluster.NumWorkers = number_of_workers;

% Create pool of workers on cluster
local_pool = parpool(local_cluster, number_of_workers);


%% END OF SCRIPT