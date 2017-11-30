function [] = Setup()

% config = frame_config('','', 'compile');
config.matconvv_path = '../../matconvnet-1.0-beta25/';
cuda_root = ''; 
% matconvv can automatically guess cuda dir, if not
% please specify the cuda directory

current_dir = pwd();

cd(fullfile(config.matconvv_path, 'matlab'));

if ispc
    cuda_method = 'nvcc';
else
    cuda_method = 'mex';
end

if isempty(cuda_root)
    vl_compilenn('EnableGPU', false);
    %, 'CudaMethod', 'nvcc');
else
    vl_compilenn('EnableGPU', false, 'CudaRoot', cuda_root, ...
        'CudaMethod', cuda_method);
end

cd(current_dir);