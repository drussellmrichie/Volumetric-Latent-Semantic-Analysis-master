% This file reads in files for each type's token x context matrix, extracts
% a convex hull, and calculates its volume.
% 
% BE AWARE THAT MATLAB ALPHABETIZES FILES IN A WEIRD WAY, meaning you have
% to number the files 00, 01, ... 98, 99.
%
% See http://en.wikipedia.org/wiki/Conceptual_Spaces for leads on potential
% justification that natural categories (but...maybe not necessarily
% linguistic categories?) are convex regions???
%%
%cd /Users/russellrichie/introcompling/FinalProject/token_by_context_files/test;
%cd /Users/russellrichie/introcompling/FinalProject/token_by_context_files_with_stops;
%cd /Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files;
%cd /Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_with_stops;
%cd /Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_without_stops_with_one_svd;
%cd /Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_with_stops_with_one_svd_not_lemmatized;
%cd /Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_with_stops_with_one_svd_not_lemmatized_first_svd_dim_kept/;
%cd /Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_without_stops_with_one_svd_not_lemmatized_first_svd_dim_discarded/;
cd /Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_without_stops_with_one_svd_lemmatized_first_svd_dim_discarded/;

clear;

files = dir('*.txt');

file_n = 0;

token_total = 500;

volumes = zeros(1,length(files));
%%
for file = files'
    clear K v infile matrix trunc_matrix;
    file_n = file_n + 1
    infile = file.name
    matrix = dlmread(infile);
    matrix = matrix(:,2:8); %comment this off if keeping first dim
    %[U,S,V] = svd(matrix);
    %trunc_matrix = matrix(1:token_total,:); % this only takes the first token_total-th tokens 
    %trunc_matrix = U(:,1:8);
    %trunc_matrix = U(:,2:9); %remove first SVD column since it's apparently junk
    [K, v] = convhulln(matrix);
    volumes(file_n) = v;
end

cd /Users/russellrichie/introcompling/FinalProject/BNC/;