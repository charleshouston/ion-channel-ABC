clc
clearvars

% Script to obtain time constants from i_CaL current traces
inact0 = csvread("Fig3A_trace_inact_0.csv");
inact10 = csvread("Fig3A_trace_inact_10.csv");
inact20 = csvread("Fig3A_trace_inact_20.csv");

t_inact0 = inact0(:,1);I_inact0 = inact0(:,2);
t_inact10 = inact10(:,1);I_inact10 = inact10(:,2);
t_inact20 = inact20(:,1);I_inact20 = inact20(:,2);

t_inact0 = t_inact0-t_inact0(1);
t_inact10 = t_inact10-t_inact10(1);
t_inact20 = t_inact20-t_inact20(1);

I_inact0 = I_inact0/max(I_inact0);
I_inact10 = I_inact10/max(I_inact10);
I_inact20 = I_inact20/max(I_inact20);

[fit_inact0, gof_inact0] = fit(t_inact0,1-I_inact0,'exp1');
[fit_inact10, gof_inact10] = fit(t_inact10,1-I_inact10,'exp1');
[fit_inact20, gof_inact20] = fit(t_inact20,1-I_inact20,'exp1');

V = [0,10,20]
fit = [fit_inact0.b,fit_inact10.b,fit_inact20.b];
gof = [gof_inact0.rmse,gof_inact10.rmse,gof_inact20.rmse];
tau = -1./fit
sd = abs(-fit.^-2 .* gof)