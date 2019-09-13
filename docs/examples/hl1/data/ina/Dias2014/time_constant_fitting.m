clc
clearvars

% Script to obtain time constants from i_CaL current traces
inactn30 = csvread("Fig6_trace_inact_n30.csv");
inactn40 = csvread("Fig6_trace_inact_n40.csv");
inactn50 = csvread("Fig6_trace_inact_n50.csv");
inactn60 = csvread("Fig6_trace_inact_n60.csv");

t_inactn30 = inactn30(:,1);I_inactn30 = inactn30(:,2);
t_inactn40 = inactn40(:,1);I_inactn40 = inactn40(:,2);
t_inactn50 = inactn50(:,1);I_inactn50 = inactn50(:,2);
t_inactn60 = inactn60(:,1);I_inactn60 = inactn60(:,2);

t_inactn30 = t_inactn30-t_inactn30(1);
t_inactn40 = t_inactn40-t_inactn40(1);
t_inactn50 = t_inactn50-t_inactn50(1);
t_inactn60 = t_inactn60-t_inactn60(1);

I_inactn30 = I_inactn30/max(I_inactn30);
I_inactn40 = I_inactn40/max(I_inactn40);
I_inactn50 = I_inactn50/max(I_inactn50);
I_inactn60 = I_inactn60/max(I_inactn60);

[fit_inactn30,gofn30] = fit(t_inactn30,1-I_inactn30,'exp1');
[fit_inactn40,gofn40] = fit(t_inactn40,1-I_inactn40,'exp1');
[fit_inactn50,gofn50] = fit(t_inactn50,1-I_inactn50,'exp1');
[fit_inactn60,gofn60] = fit(t_inactn60,1-I_inactn60,'exp1');

V = [-60,-50,-40,-30]
fit = [fit_inactn60.b,fit_inactn50.b,fit_inactn40.b,fit_inactn30.b];
gof = [gofn60.rmse,gofn50.rmse,gofn40.rmse,gofn30.rmse];

tau = -1./fit
sd = abs(-fit.^-2 .* gof)