load('frap.mat')
x = im2double(experiment.postbleach.image_data);

%%
f = fopen("/home/sms/Documents/MATLAB/frappe/x.bin");
X = fread(f, "float32");
fclose(f);

X = reshape(X, 256, 256, 110, 5);
x = X(:, :, :, 5);
%%
frame = reshape(x(:, :, 1), 256, 256);
imagesc(frame)
colorbar
caxis([-0.5, 1])
%%
normal = normalize(frame(:));
normal = reshape(normal, 256, 256);
imagesc(normal)
colorbar
%%
frame_noisy = frame + sqrt(0.05) * randn(size(frame));
imagesc(frame_noisy)
colorbar
caxis([-0.5, 1])
%%
Y = fft2(frame, 256+8, 256);
Z = log(1+Y);
figure(1)
imagesc(abs(fftshift(Z)))
colorbar
figure(2)
imagesc(real(fftshift(Z)))
colorbar
figure(3)
imagesc(imag(fftshift(Z)))
colorbar
caxis([0, 10])

%%
inertia = zeros(300, 1);
load("frap.mat")
x = im2double(experiment.postbleach.image_data);
for t = 1:300
    for i = 1:256
        for j= 1:256
            inertia(t) = inertia(t) + ((i-128)^2+(j-128)^2)*x(i, j, t);
        end
    end
end
figure(1)
plot(1:300, inertia/(max(inertia)))
hold on

%%
inertia = zeros(300, 1);
mask = im2double(experiment.bleach.image_data(:,:,2));
newmask = padarray(mask, 10, 0, 'pre');
for t = 1:300
    for i = 1:256
        for j= 1:256
            inertia(t) = inertia(t) + newmask(i,j)*x(i, j, t);
        end
    end
end
figure(1)
plot(1:300, inertia/(max(inertia)))
hold off

%%

f = fopen("/home/sms/Documents/MATLAB/frappe/first.bin");
first = fread(f, "float32");
fclose(f);

f = fopen("/home/sms/Documents/MATLAB/frappe/second.bin");
second = fread(f, "float32");
fclose(f);

f = fopen("/home/sms/Documents/MATLAB/frappe/varval.bin");
varval = fread(f, "float32");
fclose(f);

%%
first = reshape(first, 110, 5);
second = reshape(second, 110, 5);
varval = reshape(varval, 110, 5);

%%
figure(1)
plot(first)
figure(2)
plot(second)
figure(3)
plot(varval(11:end,:))
figure(4)
plot(varval(1:10,:))

%%

k = fftshift(fftshift(fft2(x),1),2);
Z = log(1+abs(k));
implay(Z)

%%
k = fft2(x);
Z = log(1+abs(k));
%Z = fftshift(Z);
implay(Z)

%%
k       = log(1+fft2(x));
Z       = abs(fftshift(k));
implay(Z)

%%
lower_bound = log10(0.001); upper_bound = log10(0.1);
data = 10.^(lower_bound + (upper_bound-lower_bound) * rand(1,100000));