% MATLAB program to kill a parallel pool,
% used for interaction with the MATLAB engine in python.
% Precedes exit from MATLAB session.

%%
% Kill pool
delete(gcp)

%% END OF SCRIPT