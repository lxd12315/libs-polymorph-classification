% Copy to local_config.m and edit paths locally. Do not commit private data paths.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
dataFile = fullfile(projectRoot, 'data', 'classification', 'julei.mat');
wavelengthFile = fullfile(projectRoot, 'data', 'classification', 'w.mat');

assert(isfile(dataFile), 'Missing private spectral data: %s', dataFile);
assert(isfile(wavelengthFile), 'Missing wavelength data: %s', wavelengthFile);

addpath(fullfile(projectRoot, 'src'));
train_validate_modern_classifier(projectRoot);
