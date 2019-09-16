%% Function to read python file to MATLAB
function [a] = loadpickle(filename)
    if ~exist(filename,'file')
        error('%s is not a file',filename);
    end
    outname = ['tempname' , '.mat'];
    %
    pyscript = ['import cPickle as pickle;import sys;import scipy.io;file=open(''', filename, ''',''r'');dat=pickle.load(file);file.close();scipy.io.savemat(''', outname, ''',mdict={''STRUCTURES_data'': dat})'];
%     pyscript = ['import cPickle as pickle;import sys;import numpy;import scipy.io;file=open(''', filename ,''',''r'');dat=pickle.load(file);file.close();scipy.io.savemat(''' , outname , ''',mdict={''RVEs_postproc'': RVE_postproc})'];
%     system(['python -c ''' pyscript '''']);
    system(['python -c "', pyscript, '"']);
    %
    % Old commands:
%     pyscript = ['import cPickle as pickle;import sys;import scipy.io;file=open("' filename '","r");dat=pickle.load(file);file.close();scipy.io.savemat("' outname '",mdict={"STRUCTURES_data": dat})'];
% %     pyscript = ['import cPickle as pickle;import sys;import numpy;import scipy.io;file=open(''', filename ,''',''r'');dat=pickle.load(file);file.close();scipy.io.savemat(''' , outname , ''',mdict={''RVEs_postproc'': RVE_postproc})'];
%     system(['python -c ''' pyscript '''']);
    %
%     system(['LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/mkl/lib/intel64:/opt/intel/composer_xe_2013/lib/intel64;python -c ''' pyscript '''']);
%     system(['LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2016/linux/mkl/lib/intel64:/opt/intel/compilers_and_libraries_2016/linux/lib/intel64;python -c ''' pyscript '''']);
    a = load(outname);
end
