% File       : show_image.m
% Created    : Thu Nov 02 2017 05:00:51 PM (+0100)
% Description: Show gray-scale image from binary file
% Copyright 2017 ETH Zurich. All Rights Reserved.
clear('all');

filename = 'elvis.50.txt'

A = load(filename);
imshow(A);
