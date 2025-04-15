function dbi = daviesbouldin(X, labels)
    % Number of clusters
    K = max(labels);
    
    % Compute the centroids of clusters
    centroids = arrayfun(@(k) mean(X(labels == k, :), 1), 1:K, 'UniformOutput', false);
    centroids = cat(1, centroids{:});
    
    % Compute the cluster dispersions
    S = arrayfun(@(k) mean(pdist2(X(labels == k, :), centroids(k, :))), 1:K);
    
    % Compute the Davies-Bouldin Index
    Rij = zeros(K);
    for i = 1:K
        for j = 1:K
            if i ~= j
                Rij(i, j) = (S(i) + S(j)) / pdist2(centroids(i, :), centroids(j, :));
            end
        end
    end
    
    dbi = mean(max(Rij, [], 2));
end