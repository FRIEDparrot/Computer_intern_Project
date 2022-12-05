% this function would use numbers to code different species
function out = coding_species(arr)

% firstly we create a name cell to storage the species
species_arr = ["Init"];
species_arr(1) = cell2mat(arr(1));
% ****initialize the string array as the first species element******

species_rec = zeros(length(arr),1);  % encode and record species into number
j = 0;
for i = 1: length(arr)   % note that the cell type also support index
    if ~ismember( string(arr(i)), species_arr )
        species_arr = [species_arr , arr(i)];
        % append the new species in the species array
    end
    % then judge the type of species and encode it into numbers
    species_rec(i) = find(species_arr == string(arr(i)));  
    % find the index of the corresponding species as the encode number
    % the find function is used for searching the index it is
end
out  = species_rec;

end
