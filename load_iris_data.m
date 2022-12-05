function out = load_iris_data()

data_iris = readmatrix("iris.csv","Range","B2:E151");
species_iris = readcell("iris.csv","Range","F2:F151");

species_iris_code = coding_species(species_iris) -1;
% start the species from the index 0
out = [data_iris,species_iris_code];
% concatnate the matrix and the array together
end 