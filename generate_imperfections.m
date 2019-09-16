function [Sample] = generate_imperfections(UserSettings_UQlab)
uqlab
%
%Define input properties (amplitudes of Mode 1 and Mode 2 imperfections)
Input.Marginals(1).Type = UserSettings_UQlab.distribution;
% Input.Marginals(1).Type = 'Lognormal';
%Input.Marginals(1).Type = 'Gaussian';

Input.Marginals(1).Moments = UserSettings_UQlab.Moments;

% Alternatively, you can give the PARAMETERS of the distribution directly,
%instead of the MOMENTS. If that's what you want, comment line above and
%uncomment the next one.
% Input.Marginals(1).Parameters = UserSettings_UQlab.Parameters;

Input.Copula.Type = 'Independent';

%Create Input object
Imperfections = uq_createInput(Input) ;

%print object properties
uq_print(Imperfections)

%% SHOW THE DISTRibution
% uq_display(Imperfections)

%% Create diffrent sampling sizes n vector
Sample = uq_getSample(UserSettings_UQlab.samplesize,'LHS');
%
%
end

