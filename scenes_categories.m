function [hbn_tree, data] = scenes_categories()
    data = get_data();
    hbn_tree = build_hbn_tree(data);
    hbn_tree = sample_hbn_tree(hbn_tree, data);
end

%% One (few)-shot(s) learning for fixed tree structure

function b_cat = sample_category_most_likely_cat_id(hbn_tree, x)
    posteriors = zeros(1, numel(hbn_tree.z_s));
    for c=1:numel(hbn_tree.z_s)
        posteriors(c) = exp(sample_category_loglikelihood(hbn_tree, c, x))*...
            category_ncrp_prior(hbn_tree, c);
    end
    [~, b_cat] = max(posteriors);
end

function probabilities = sample_category_probabilities(hbn_tree, x)
    posteriors = zeros(1, numel(hbn_tree.z_s));
    for c=1:numel(hbn_tree.z_s)
        posteriors(c) = exp(sample_category_loglikelihood(hbn_tree, c, x))*...
            category_ncrp_prior(hbn_tree, c, x);
    end
    probabilities = posteriors/sum(posteriors);
end

function log_likelihood = sample_category_loglikelihood(hbn_tree, c, x)
    mu = hbn_tree.level_1.mu_c(c, :);
    tau = hbn_tree.level_1.tau_c(c, :);
    C = 0;
    log_likelihood = 0.5*sum(log(tau))-0.5*sum(tau.*(x-mu).^2)+C;
end

function ncrp = category_ncrp_prior(hbn_tree, c)
    k = hbn_tree.z_s(c);
    k_basic_nodes = numel(find(hbn_tree.z_s == k));
    total_basic_nodes = numel(hbn_tree.z_s);
    first = crp_prior(k_basic_nodes, total_basic_nodes, hbn_tree.gamma);
    
    c_samples = numel(find(hbn_tree.z_b == c));
    total_samples = numel(hbn_tree.z_b);
    second = crp_prior(c_samples, total_samples, hbn_tree.gamma);
    
    ncrp = first*second;
end

function crp = crp_prior(n_k, n, gamma)
    crp = n_k/(n-1+gamma);
end
%% Tree samplers

function hbn_tree = build_hbn_tree(data)
    fprintf('Building the HBN classifier...\n');
    
    % Initializing the tree
    hbn_tree = struct;
    
    % Setting fixed level-wide prior parameters (values taken from the
    % paper)
    hbn_tree.a0 = 1;
    hbn_tree.b0 = 1;
    hbn_tree.t = 3;
    hbn_tree.nu = 1;
    hbn_tree.gamma = gamraternd(1, 1);
    
    % Setting feature dimensionality
    hbn_tree.D = size(data.X, 2);
    
    if (numel(data.z_b) && numel(data.z_s))
        % In case we have pre-specified the structure of the tree
        hbn_tree.z_b = data.z_b;
        hbn_tree.z_s = data.z_s;
    else
        % No predefined structure, define its primer here
        % ???
        hbn_tree.z_b = [];
        hbn_tree.z_s = [];
    end
            
    % Sampling parameters on all 3 levels using the initial top-down
    % approach (without conditioning on data)
    init = true;
    hbn_tree = sample_level_3(hbn_tree, init);
    hbn_tree = sample_level_2(hbn_tree, init);
    hbn_tree = sample_level_1(hbn_tree, data, init);
    fprintf('HBN classifier built\n\n');
end

function hbn_tree = sample_hbn_tree(hbn_tree, data)
    % We want the actual samples now, not the initial values from the prior
    init = false;
    Gibbs_steps = 200;
    fprintf('Starting Gibbs sampler...\n');
    for step = 1:Gibbs_steps
        sample_level_1(hbn_tree, data, init);
        sample_level_2(hbn_tree, init);
        sample_level_3(hbn_tree, init);
        fprintf('Completed %d/%d Gibbs sweeps\n', step, Gibbs_steps);
    end
    fprintf('Parameters should be inferred now\n');
end

function hbn_tree = sample_level_3(hbn_tree, init)
    if (init)
        hbn_tree.level_3 = struct;
        hbn_tree.level_3.alpha_0 = gamraternd(1,1);
        hbn_tree.level_3.tau_0 = gamraternd(1,1);
    else
        % Total number of alpha_k^d variables on level 2:
        % KD = K (number of nodes) * D (vectors dimensionality)
        KD = numel(hbn_tree.z_s)*hbn_tree.D;
        hbn_tree.level_3.alpha_0 = gamraternd(1+KD,...
            1+KD*mean(hbn_tree.level_2.alpha_k(:)));
        hbn_tree.level_3.tau_0 = gamraternd(1+KD, ...
            1+sum(hbn_tree.level_2.mu_k(:).^2));
    end
end

function hbn_tree = sample_level_2(hbn_tree, init)
    if (init)
        hbn_tree.level_2 = struct;
        hbn_tree.level_2.mu_k = normprecrnd(0, 1./hbn_tree.level_3.tau_0, ...
            max(hbn_tree.z_s), hbn_tree.D);
        hbn_tree.level_2.alpha_k = exprnd(hbn_tree.level_3.alpha_0,...
            max(hbn_tree.z_s), hbn_tree.D);
        hbn_tree.level_2.tau_k = gaminvrnd(hbn_tree.a0, hbn_tree.b0, ...
            max(hbn_tree.z_s), hbn_tree.D);
    else
        for k=1:max(hbn_tree.z_s)
            basic_nodes_children = find(hbn_tree.z_s == k);
            children_tau = hbn_tree.level_1.tau_c(basic_nodes_children, :);
            children_mu = hbn_tree.level_1.mu_c(basic_nodes_children, :);
            
            norm_prec = (1./hbn_tree.level_3.tau_0+sum(1./(hbn_tree.nu.*...
                children_tau)));
            norm_mean = sum(children_mu./(hbn_tree.nu.*children_tau))./...
                norm_prec;
            
            hbn_tree.level_2.mu_k(k, :) = normprecrnd(norm_mean, norm_prec, ...
                1, hbn_tree.D);
            
            gaminv_alpha = hbn_tree.a0+numel(basic_nodes_children).*...
                hbn_tree.level_2.alpha_k(k, :);
            gaminv_beta = hbn_tree.b0+hbn_tree.level_2.alpha_k(k, :).*sum(...
                children_tau);
            
            hbn_tree.level_2.tau_k(k, :) = gaminvrnd(gaminv_alpha, gaminv_beta, ...
                1, hbn_tree.D);
            
            % Sample tau_d^k according to the formula and new alpha_k
                        % MH-sample alpha_k first
            alpha_k_prop = gamraternd(hbn_tree.t, hbn_tree.t./hbn_tree.level_2.alpha_k(k, :),...
                1, hbn_tree.D);
            prob_prop = level_2_alpha_cond_prob(alpha_k_prop, hbn_tree.level_3.alpha_0, ...
                children_tau, hbn_tree.level_2.tau_k(k, :));
            prob_prev = level_2_alpha_cond_prob(hbn_tree.level_2.alpha_k(k, :), hbn_tree.level_3.alpha_0, ...
                children_tau, hbn_tree.level_2.tau_k(k, :));
            accept = min(prob_prop./prob_prev, 1);
            assign_new_values = rand(1, numel(accept)) > 1-accept;
            ind_to_change = find(assign_new_values == 1);
            hbn_tree.level_2.alpha_k(k, ind_to_change) = alpha_k_prop(ind_to_change);
        end
    end
end

function prob = level_2_alpha_cond_prob(alpha_k, alpha_0, tau_children, tau_k)
    S_k = sum(tau_children);
    T_k = sum(log(tau_children));
    n_k = size(tau_children, 1);
    exp_mult = (alpha_0+...
        S_k./tau_k-T_k);
    exp_overall = exp(-alpha_k.*exp_mult);
    top_term = (alpha_k./tau_k).^(alpha_k.*n_k);
    bottom_term = gamma(alpha_k).^n_k;
    prob = exp_overall.*top_term./bottom_term;
end

function hbn_tree = sample_level_1(hbn_tree, data, init)
    if (init)
        hbn_tree.level_1 = struct;
        for c=1:numel(hbn_tree.z_s)
            k = hbn_tree.z_s(c);
            mu = hbn_tree.level_2.mu_k(k, :);
            nu = hbn_tree.nu;
            alpha = hbn_tree.level_2.alpha_k(k, :);
            beta = hbn_tree.level_2.alpha_k(k, :)./...
                hbn_tree.level_2.tau_k(k, :);
            
            [hbn_tree.level_1.mu_c(c, :), hbn_tree.level_1.tau_c(c, :)] = ...
                normgamrnd(mu, nu, alpha, beta, 1, hbn_tree.D);
        end
    else
        for c=1:numel(hbn_tree.z_s)
            k = hbn_tree.z_s(c);
            X_chunk = data.X(find(data.z_b == c));
            X_mean = mean(X_chunk);
            n = size(X_chunk, 1);
            
            alpha = hbn_tree.level_2.alpha_k(k, :)+n/2;
            beta = hbn_tree.level_2.alpha_k(k, :)./...
                hbn_tree.level_2.tau_k(k, :)+0.5*sum((X_chunk-repmat(X_mean, ...
                size(X_chunk, 1), 1)).^2)+0.5*(n*hbn_tree.nu/(hbn_tree.nu+n))*(X_mean-...
                hbn_tree.level_2.mu_k(k, :)).^2;
            mu = (hbn_tree.nu.*hbn_tree.level_2.mu_k(k, :)+n.*X_mean)/...
                (hbn_tree.nu+n);
            nu = hbn_tree.nu+n;
            [hbn_tree.level_1.mu_c(c, :), hbn_tree.level_1.tau_c(c, :)] = ...
                normgamrnd(mu, nu, alpha, beta, 1, hbn_tree.D);
        end
    end
end

%% Custom distributions

% Normal distribution (with precision instead of MATLAB's default variance)
function sample = normprecrnd(mu, tau, varargin)
    sigma = sqrt(1./tau);
    sample = normrnd(mu, sigma, varargin{:});
end

% Gamma distribution (with rate instead of MATLAB's default scale)
function sample = gamraternd(alpha, beta, varargin)
    sample = gamrnd(alpha, 1./beta, varargin{:});
end

% Inverse Gamma distribution
function sample = gaminvrnd(alpha, beta, varargin)
    sample = 1./gamraternd(alpha, 1./beta, varargin{:});
end

% Normal Gamma distribution
function [mu, tau] = normgamrnd(mu, nu, alpha, beta, varargin)
    tau = gamraternd(alpha, beta, varargin{:});
    mu = normprecrnd(mu, 1./(nu.*tau), varargin{:});
end

%% Data loaders

function data = get_data()
    data_path = fullfile('scenes_hbn_data_validation.mat');
    reload_data = false;
    if (~exist(data_path, 'file') || reload_data)
        data = load_data(data_path);
    else
        load(data_path, 'data');
        fprintf('Data has been loaded from %s\n\n', data_path);
    end
end

function data = load_data(saved_data_path)
    % Load both the input vectors X and their caterogies z_s and z_b
    fprintf('Loading input data...\n');
    matconvnet_path = '/Users/Sergey/matconvnet/matlab';
    addpath(matconvnet_path);
    try 
        vl_setupnn;
    catch
        error('Could not find matconvnet. Please check matconvnet_path.');
    end
    project_path = '/Users/Sergey/cv_project';
    data_path = fullfile(project_path, 'data/images/train');
    net_path = fullfile(project_path, 'miniplacesCNN/net-aug-epoch-36.mat');
    imstats_path = fullfile(project_path, 'miniplacesCNN/imageStats.mat');
    
    load(net_path, 'net') ;
    load(imstats_path, 'rgbCovariance');
    
    % Specify categories as subpaths to directories with their images
    categories = {'/b/bar', '/f/food_court', '/r/restaurant', ...
        '/c/conference_room', '/c/classroom', ...
        '/b/bamboo_forest', ...
        '/b/badlands', '/c/canyon', '/m/mountain'};
    super_categories = [1 1 1 2 2 3 4 4 4];
    samples_per_categories = 900*ones(1, numel(categories));
    
    % Limiting the number of samples for one of the categories to test the
    % generalization later on
    samples_per_categories(3) = 5;
    
    % Modification for the shorter tree (for tests)
    % Uncomment to test and adjust the sampling process
    tree_test = false;
    if (tree_test)
        categories = categories(1:2);
        super_categories = super_categories(1:2);
        samples_per_categories = 10*ones(1, numel(categories));
    end
    
    data.X = [];
    data.names = {};
    data.z_b = [];
    data.z_s = super_categories;
    
    cat_ind = 1;
    for cat = categories
        fprintf('Loading category %s from %s...\n', ...
            cell2mat(cat), fullfile(data_path, cell2mat(cat)));
        cat_images = dir(fullfile(data_path, cell2mat(cat), '*.jpg'));
        for cat_image_idx = 1:numel(cat_images)
            image_path = fullfile(data_path, cell2mat(cat), ...
                cat_images(cat_image_idx).name);
            features = get_CNN_features(net, image_path, rgbCovariance);
            data.names{end+1} = image_path;
            data.X = [data.X; features'];
            data.z_b = [data.z_b cat_ind];
            
            if (cat_image_idx >= samples_per_categories(cat_ind))
                break;
            end
        end
        fprintf('Category %s loaded [%d/%d]\n', cell2mat(cat), cat_ind,...
            numel(categories));
        cat_ind = cat_ind + 1;
    end
   
    fprintf('Normalizing the data...\n');
    % Normalizing the mean and variance of the input data
    data.X = data.X - mean(data.X(:));
    data.X = data.X/std(data.X(:));
    
    % Saving the data to restore it if it is needed
    save(saved_data_path, 'data');
    fprintf('Input data has been saved to %s\n\n', saved_data_path);
end

function features = get_CNN_features(net, im_path, rgbCov)
    % This function uses the preloaded model to generate feature vectors
    % with the help of augmented dataset. It takes an image path, loads an
    % image, crops and flips it multiple times and then averages the score
    % on a particular layer
        
    if (~isfield(net.normalization, 'numAugments'))
        net.normalization.numAugments = 2;
    end
    
    [v,d] = eig(rgbCov) ;
    rgbVariance = 0.1*sqrt(d)*v';
    averageImage = net.normalization.averageImage;
    if numel(averageImage) == 3
      averageImage = reshape(averageImage, 1,1,3) ;
    end
    
    % Changing the type of last layer from softmaxloss to softmax
    net.layers{1,end}.type = 'softmax';
    net.layers{1,end}.name = 'prob';

    imo = zeros(net.normalization.imageSize(1), net.normalization.imageSize(2), 3, ...
                net.normalization.numAugments, 'single') ;

    imt = imread(im_path) ;
    
    % Use this if you want to display the image
    % Display im_resize using imagesc function:
    % im_resize = imresize(imt, net.normalization.imageSize(1:2)) ;
    
    imt = single(imt) ;

    % Resize the image
    w = size(imt,2) ;
    h = size(imt,1) ;
    factor = [(net.normalization.imageSize(1)+net.normalization.border(1))/h ...
            (net.normalization.imageSize(2)+net.normalization.border(2))/w];

    if any(abs(factor - 1) > 0.0001)
        imt = imresize(imt, ...
                   'scale', factor, ...
                   'method', net.normalization.interpolation) ;
    end

    % Data augmentation parameters
    lim_ratio = 0.8;
    n_sizes = 2;
    w_for_size = 2;
    
    si = 1 ;
    scores = [];
    % Crop and flip the image
    for ai = 1:(net.normalization.numAugments/2)
      if (ai == 1)
          size_factor = 1.0;
      else
          size_factor = 1.0-(ceil((ai-1)/w_for_size))*(1.0-lim_ratio)/n_sizes;
      end
      sz = round(min(net.normalization.imageSize(1:2)' .* size_factor, [w;h])) ;
      dx = randi(w - sz(2) + 1, 1) ;
      dy = randi(h - sz(1) + 1, 1) ;

      sx = round(linspace(dx, sz(2)+dx-1, net.normalization.imageSize(2))) ;
      sy = round(linspace(dy, sz(1)+dy-1, net.normalization.imageSize(1))) ;

      sx_fl = fliplr(sx);
      if ~isempty(averageImage)
        offset = averageImage ;
        if ~isempty(rgbVariance)
            offset = bsxfun(@plus, offset, reshape(rgbVariance * randn(3,1), 1,1,3)) ;
        end
        imo(:,:,:,si) = bsxfun(@minus, imt(sy,sx,:), offset) ;
        imo(:,:,:,si+1) = bsxfun(@minus, imt(sy,sx_fl,:), offset) ;
      else
        imo(:,:,:,si) = imt(sy,sx,:) ;
        imo(:,:,:,si+1) = imt(sy,sx_fl,:) ;
      end
      res_1 = vl_simplenn(net, imo(:,:,:,si)) ;
      res_2 = vl_simplenn(net, imo(:,:,:,si+1)) ;
      
      % The particular model that I have used has 4 convolutional layers
      % + 1 last FC + softmax
      %
      % Because of the way matconvnet builds networks, reLU, maxpooling
      % and dropout are considered separate layers. The last 'truly'
      % convolutional layer has the layer_number index in the model
      
      layer_number = 13;
      new_scores = squeeze(gather(res_1(layer_number).x))+...
          squeeze(gather(res_2(layer_number).x));
      
      if (numel(scores) == 0)
        scores = new_scores;
      else
        scores = scores+new_scores;
      end
      si = si + 2;
    end

    features = scores./net.normalization.numAugments;
end