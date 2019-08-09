clc
clearvars

% Script to obtain time constants from i_CaL current traces
inact0 = csvread("Fig7_trace_inact0.csv");
inactn10 = csvread("Fig7_trace_inactn10.csv");
inactn20 = csvread("Fig7_trace_inactn20.csv");

t_inact0 = inact0(:,1);I_inact0 = inact0(:,2);
t_inactn10 = inactn10(:,1);I_inactn10 = inactn10(:,2);
t_inactn20 = inactn20(:,1);I_inactn20 = inactn20(:,2);

t_inact0 = t_inact0-t_inact0(1);
t_inactn10 = t_inactn10-t_inactn10(1);
t_inactn20 = t_inactn20-t_inactn20(1);

I_inact0 = I_inact0/max(I_inact0);
I_inactn10 = I_inactn10/max(I_inactn10);
I_inactn20 = I_inactn20/max(I_inactn20);

fit_inact0 = fit(t_inact0,1-I_inact0,'exp1');
fit_inactn10 = fit(t_inactn10,1-I_inactn10,'exp1');
fit_inactn20 = fit(t_inactn20,1-I_inactn20,'exp1');

V = [-20,-10,0]
tau = -1./[fit_inactn20.b,fit_inactn10.b,fit_inact0.b]