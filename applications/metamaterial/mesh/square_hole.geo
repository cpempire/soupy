SetFactory("OpenCASCADE");
//Mesh.Algorithm = 6;
//Mesh.CharacteristicLengthMin = 1/4;
//Mesh.CharacteristicLengthMax = 1/4;

Rectangle(1) = {-1, -1, 0, 12, 12, 0};
Disk(2) = {5, 5, 0, 3};
Disk(3) = {5, 5, 0, 1};
BooleanDifference{Surface{1:2};Delete;}{Surface{3};Delete;}
//BooleanFragments{Surface{1:3}; Delete;}{}

