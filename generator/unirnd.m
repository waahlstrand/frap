function r = unirnd(lower_bound, upper_bound, n)

r = lower_bound + (upper_bound-lower_bound) * rand(1, n);

end
