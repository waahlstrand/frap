function r = logunirnd(lower_bound, upper_bound, n)

lower_bound = log10(lower_bound); upper_bound = log10(upper_bound);
r = 10.^(lower_bound + (upper_bound-lower_bound) * rand(1, n));

end

