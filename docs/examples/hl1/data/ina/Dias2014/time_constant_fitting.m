clc
clearvars

% Script to obtain time constants from i_Na current traces
peak_act = csvread("peak_activation.csv");
peak_inact = csvread("peak_inactivation.csv");

t_act = peak_act(:,1); I_act = peak_act(:,2);
t_inact = peak_inact(:,1); I_inact = peak_inact(:,2);

t_act = t_act-t_act(1);
t_inact = t_inact-t_inact(1);

I_act = -(I_act-I_act(1));
I_act = (I_act-min(I_act))/(max(I_act)-min(I_act));
I_inact = 1-I_inact;
I_inact = (I_inact-min(I_inact))/(max(I_inact)-min(I_inact));

act_model = fittype('(1-exp(-t/tau))^3','independent','t');
inact_model = fittype('exp(-t/tau)','independent','t');

[fit_act,gof_act] = fit(t_act, I_act, act_model, 'Start', [0.5]);
[fit_inact,gof_inact] = fit(t_inact,I_inact,inact_model,'Start',[5]);
