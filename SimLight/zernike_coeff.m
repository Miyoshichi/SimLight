function coeffs = zernike_coeff(wavefront, wavelength, n, m)
%
% Return the Zernike coefficients of wavefront of a light field.
%
% @since 1.0.0
% @param {array} [wavefront] Wavefront for Zernike coefficients calculation.
% @param {float} [wavelength] Wavelength of source.
% @param {array} [n] Zernike order.
% @param {array} [m] Angular frequency.
% @return {array} [coeffs] Zernike coefficients.
% @see dependencies
%

    wavelength = wavelength * 1e6;
    k = - 2 * pi / wavelength;

    L = size(wavefront, 1);
    x = -1 : 2 / (L - 1) : 1;
    [X, Y] = meshgrid(x);
    X = X(:);
    Y = Y(:);
    [theta, R] = cart2pol(X, Y);

    W = wavefront(:);
    is_in_circle = zeros(L^2, 1);
    for i = 1 : L^2
        if isnan(W(i)) || R(i) > 1
            is_in_circle(i) = 0;
        else
            is_in_circle(i) = 1;
        end
    end

    is_in_circle = logical(is_in_circle);
    Z = zernike_fun(n, m, R(is_in_circle), theta(is_in_circle), 'norm');
    coeffs = Z \ W(is_in_circle) / k;

end
