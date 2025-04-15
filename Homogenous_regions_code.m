close all
clear all
%% Generating dates from 01-jan-1901 to 31-12-2021
load("Input_data.mat")
% Seperating the data for jan to may
k=1;st_month=1;end_month=5;
for z = st_month:end_month
    for i = min(mons_ppt_data(:,1)):max(mons_ppt_data(:,1))
        jm_ppt_data{k,1}=mons_ppt_data(find(mons_ppt_data(:,2)==z & mons_ppt_data(:,1)==i),3:end);
        k=k+1;
    end
end
for i = 1:((max(mons_ppt_data(:,1))-(min(mons_ppt_data(:,1)))))+1
    for j = 1:size(jm_ppt_data,2)
        jmyr_ppt_data(i,1)={cat(1,jm_ppt_data{i:(((max(mons_ppt_data(:,1))-(min(mons_ppt_data(:,1)))+1))):size(jm_ppt_data,1),j})};
    end
end

% Seperating the data for june to december
k=1;st_month=6;end_month=12;
for z = st_month:end_month
    for i = min(mons_ppt_data(:,1)):max(mons_ppt_data(:,1))
        jud_ppt_data{k,1}=mons_ppt_data(find(mons_ppt_data(:,2)==z & mons_ppt_data(:,1)==i),3:end);
        k=k+1;
    end
end
for i = 1:((max(mons_ppt_data(:,1))-(min(mons_ppt_data(:,1)))))+1
    for j = 1:size(jud_ppt_data,2)
        judyr_ppt_data(i,1)={cat(1,jud_ppt_data{i:(((max(mons_ppt_data(:,1))-(min(mons_ppt_data(:,1)))+1))):size(jud_ppt_data,1),j})};
    end
end
jmyr_ppt_data=jmyr_ppt_data(2:end,1);
judyr_ppt_data=judyr_ppt_data(1:end-1,1);
for i = 1:length(judyr_ppt_data)
wateryrdata{i,1}=[judyr_ppt_data{i,1};jmyr_ppt_data{i,1}];
wateryr_ann_rain{i,1}=sum(wateryrdata{i,1});
end
wateryr_ann_rain=vertcat(wateryr_ann_rain{:});
wateryr_ann_rain5120=wateryr_ann_rain(51:120,:);
wateryrdata5120=wateryrdata(51:end,:);
wateryr_mean_ann_rain5120=mean(wateryr_ann_rain5120,1,'omitmissing');
tic
for j = 1:size(wateryrdata5120,2)
    for i = 1:size(wateryrdata5120,1)
        for k = 1:length(wateryrdata5120{i,j})
           [wateryr_pdtot_ent(i,k),wateryr_pdhiscon{i,j}{1,k}]=cent(wateryrdata5120{i,j}(:,k));
           [wateryr_app_totent(i,k),wateryr_app_ent{i,j}(:,k),wateryr_app_prob{i,j}(:,k)]=appent(wateryrdata5120{i,j}(:,k));
        end
    end
end
toc
wateryr_meanME5120=mean(wateryr_pdtot_ent,1,'omitnan');
wateryr_meanAE5120=mean(wateryr_app_totent,1,'omitnan');
tic
for i = 1:length(wateryrdata5120)
    wateryrcen_mean(i,:)=centri(wateryrdata5120{i,1},wateryr_ann_rain5120(i,:));
end
toc
wateryr_pdhiscon1=vertcat(wateryr_pdhiscon{:});
wateryr_mean_cent=ceil(mean(wateryrcen_mean,1,'omitNaN'));
wateryr_som_input_data=[latlon(:,2),latlon(:,1),wateryr_mean_ann_rain5120',wateryr_meanAE5120',wateryr_meanME5120',wateryr_mean_cent'];
wateryr_som_input_datatbl=array2table(wateryr_som_input_data);
wateryr_som_input_datatbl.Properties.VariableNames={'lat','long','AR','AE','ME','centroid'};


%% SOM CLustering
%% Finding the grid size
som_data=wateryr_som_input_datatbl;
som_data=table2array(som_data(:,3:6));
norm_data=normalize(som_data);
k=1;
for i = 1
    for j = 2:50
     grid_sizes{1,k}=[i j];  
     k=k+1;
    end
end
num_iterations = 1000;  % Number of iterations for training
te_values = zeros(length(grid_sizes), 1);  % Array to store topology errors
qe_values = zeros(length(grid_sizes), 1);  % Array to store quantization errors
silhouette_scores = [];
dbi_scores = [];
tic
for i = 1:length(grid_sizes)
     % Create and train the SOM with a specific topology
    grid_size = grid_sizes{i};
    net = selforgmap(grid_size, num_iterations, 3, 'hextop', 'linkdist');
    net.trainParam.epochs=1000;
    net.trainParam.lr=0.6;
    net = train(net, norm_data');  % Train SOM on your data (transpose if needed) 
    % Cluster assignment
    bmu_indices = vec2ind(net(norm_data'));
    bmu_indices_all{i}=bmu_indices;
    % Compute silhouette score
    sil_score = mean(silhouette(norm_data, bmu_indices', 'Euclidean'));
    silhouette_scores = [silhouette_scores; sil_score];
    % Compute Davies-Bouldin Index
    dbi_score = daviesbouldin(norm_data, bmu_indices);
    dbi_scores = [dbi_scores; dbi_score];

    % Calculate Quantization Error (QE)
    distances = zeros(size(norm_data, 1), 1);
    for j = 1:size(norm_data, 1)
        distances(j) = norm(norm_data(j, :)' - net.IW{1}(bmu_indices(j), :)');
    end
    qe_values(i) = mean(distances);

    % Calculate Topology Error (TE)
    % Get neuron positions on the grid
    positions = net.layers{1}.positions;
    
    % Calculate distances between input data and all neurons
    distances1 = dist(net.IW{1}, norm_data');
    
    % Sort distances to find first (BMU1) and second (BMU2) closest neurons
    [~, sorted_indices] = sort(distances1, 1);
    bmu1 = sorted_indices(1, :); % First BMU indices
    bmu2 = sorted_indices(2, :); % Second BMU indices
    
    % Compute the grid distance between BMU1 and BMU2
    grid_distances = sqrt(sum((positions(:, bmu1) - positions(:, bmu2)).^2, 1));
    
    % Topology error is the percentage of non-adjacent BMUs
    te_values(i) = mean(grid_distances > 1);

    % Display results
    fprintf('Grid Size: [%d %d], QE: %f, TE: %f\n, Silhouette Score: %f, DBI: %f\n ', ...
            grid_size(1), grid_size(2), qe_values(i), te_values(i), sil_score, dbi_score);
end
toc
error_data=[cell2mat(all_grid_sizes),qe_values,te_values,silhouette_scores,dbi_scores];

%% SOM for a particular grid size.
% The grid size is selected as [1 11] and the maximum number of iterations
% are selected as 500 and the optimal learning rate is considered as 0.6
rng(1)
data=wateryr_som_input_datatbl;
data=table2array(data(:,3:6));
% The data has to be normalized before giving it to the som
norm_data=normalize(data);
grid_size=[1 11];
num_iterations=500;
net=selforgmap(grid_size,num_iterations,3,'hextop','linkdist');
net.trainParam.lr=0.6;
net.trainParam.epochs=500;
net=train(net,norm_data');
bmu_indices=vec2ind(net(norm_data'));
center=net.IW{1};
clustered_data=[wateryr_som_input_data,bmu_indices'];
wateryr_SOM_ppt_datatbl=array2table(clustered_data);
wateryr_SOM_ppt_datatbl.Properties.VariableNames={'lat','long','MAR','MAE','MME','centroid','cluster'};


%% Cluster Data Analysis
wateryr_SOM_ppt_pointLatitude = wateryr_SOM_ppt_datatbl.lat;
wateryr_SOM_ppt_pointLongitude = wateryr_SOM_ppt_datatbl.long;
wateryr_SOM_ppt_datatbl.cluster=wateryr_SOM_ppt_datatbl.cluster;

for i = 1:length(unique(wateryr_SOM_ppt_datatbl.cluster))
    wateryr_ppt_cluster_indices{i,1}=find(wateryr_SOM_ppt_datatbl.cluster==i);
    wateryr_ppt_cluster_data{i,1}=wateryr_SOM_ppt_datatbl(wateryr_ppt_cluster_indices{i,1}(:,1),1:end-1);
    wateryr_ppt_cluster_data1{i,1}=table2array(wateryr_ppt_cluster_data{i,1}(:,3));
    wateryr_ppt_cluster_data_sort(i,1)=mean(wateryr_ppt_cluster_data1{i,1},"all");
end

indices=sort(wateryr_ppt_cluster_data_sort);
for i = 1:length(unique(wateryr_SOM_ppt_datatbl.cluster))
sorted_wateryr_ppt_cluster_data1= sort(wateryr_ppt_cluster_data_sort);
sort_indices(i,1)=find(wateryr_ppt_cluster_data_sort==indices(i));
sorted_wateryr_ppt_Cluster_indices{i,1}=wateryr_ppt_cluster_indices{sort_indices(i,1),1};
sorted_wateryr_ppt_cluster_data(i,1)=wateryr_ppt_cluster_data(sort_indices(i,1),1);
sorted_wateryr_hiscon_cluster_data{i,1}=wateryr_pdhiscon1(:,sorted_wateryr_ppt_Cluster_indices{i,1}(:,1));
sorted_wateryr_apptotent_cluster_data{i,1}=wateryr_app_totent(:,sorted_wateryr_ppt_Cluster_indices{i,1}(:,1));
sorted_wateryr_annrain_cluster_data{i,1}=wateryr_ann_rain5120(:,sorted_wateryr_ppt_Cluster_indices{i,1}(:,1));
sorted_wateryr_cent_cluster_data{i,1}=wateryrcen_mean(:,sorted_wateryr_ppt_Cluster_indices{i,1}(:,1));
sorted_wateryr_pdtotent_cluster_data{i,1}=wateryr_pdtot_ent(:,sorted_wateryr_ppt_Cluster_indices{i,1}(:,1));
end

for j = 1:length(wateryr_app_prob)
    for i = 1:length(unique(wateryr_SOM_ppt_datatbl.cluster))
        sorted_wateryr_opt_approb_cluster_data{j,i}=wateryr_app_prob{j,1}(:,sorted_wateryr_ppt_Cluster_indices{i,1}(:,1));
    end
end

wateryr_ppt_cluster_data=sorted_wateryr_ppt_cluster_data;
for i = 1:length(unique(wateryr_SOM_ppt_datatbl.cluster))
cc=ones(size(wateryr_ppt_cluster_data{i,1},1),1).*i;
wateryr_ppt_cluster_data{i,1}.cluster=cc;
end
wateryr_ppt_cluster_datatbl = vertcat(wateryr_ppt_cluster_data{:});

%% Plotting Spatial Map %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gridded plot
load('clustered_data_input.mat');
cluster=wateryr_ppt_cluster_datatbl.cluster;
clong=wateryr_ppt_cluster_datatbl.long;
clat=wateryr_ppt_cluster_datatbl.lat;
shapefilepath='E:\My_PhD\objectives\obj1\matlab_codes\India_ppt\p1\Shapefiles\India_shapefile.shp';
s=shaperead(shapefilepath);
bounds= [min(s.BoundingBox(:,1)),min(s.BoundingBox(:,2)),max(s.BoundingBox(:,1)),max(s.BoundingBox(:,2))];
res=0.25;
[xgrid,ygrid]=meshgrid(bounds(1):res:bounds(3),bounds(2):res:bounds(4));

% Define the bounding box (e.g., India's bounds)
lon_min = 67;  % Minimum longitude
lon_max = 98;  % Maximum longitude
lat_min = 5;   % Minimum latitude
lat_max = 38;  % Maximum latitude

% Define grid resolution
gridResolution = 0.25;  % 0.25° x 0.25°

% Generate grid edges
lon_edges = lon_min:gridResolution:lon_max;  % Longitude edges
lat_edges = lat_min:gridResolution:lat_max;  % Latitude edges

% Load the India shapefile
s = shaperead(shapefilepath);  % Read the shapefile

% Initialize a matrix to store clipped polygons
clippedPolygons = [];  % To store clipped grid cells

% Loop through latitudes and longitudes to create and clip grid cells
for i = 1:length(lon_edges)-1
    for j = 1:length(lat_edges)-1
        % Define vertices of the current grid cell
        xVertices = [lon_edges(i), lon_edges(i+1), lon_edges(i+1), lon_edges(i)];
        yVertices = [lat_edges(j), lat_edges(j), lat_edges(j+1), lat_edges(j+1)];
        % Compute the center of the grid cell
        centerX = mean(xVertices);
        centerY = mean(yVertices);
        
        % Check if the grid cell is inside the India shapefile boundary
        in = inpolygon(mean(xVertices), mean(yVertices), s.X, s.Y);  % Check if the grid center is inside

        if in
            distances = sqrt((clong - centerX).^2 + (clat - centerY).^2);
            [~, closestIdx] = min(distances);  % Find the nearest point
            
            % Add the grid cell to the list of clipped polygons
              % Add the grid cell and its cluster number
            clippedPolygons = [clippedPolygons; struct('x', xVertices, 'y', yVertices, ...
                                                       'Cluster', cluster(closestIdx))];
        end
    end
end

% Plot the results
figure('Position', [10, 10, 1100, 1080]);
hold on;

% Plot the India shapefile
mapshow(s, 'DisplayType', 'polygon', 'FaceColor', 'none', 'EdgeColor', 'k', 'LineWidth', 1);

% Plot the clipped grid cells filled with cluster numbers
for k = 1:length(clippedPolygons)
    fill(clippedPolygons(k).x, clippedPolygons(k).y, clippedPolygons(k).Cluster, ...
         'EdgeColor', 'none', 'LineWidth', 0.5);  % Fill color based on cluster number
end


c=colorbar;
colormap(turbo(11)); % You can use any colormap here
c = colorbar();
c.Ticks = 1:11;
c.Limits = [0.5, 11.5]; % Set limits slightly outside the range of cluster values

% Calculate positions of tick labels at the center of each color segment
tick_positions = linspace(c.Limits(1), c.Limits(2), numel(c.Ticks));
label_positions = tick_positions(1:end-1) + diff(tick_positions)/2;
c.TickLabels = {'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7', 'Cluster 8','Cluster 9','Cluster 10','Cluster 11'};
for i = 1:numel(label_positions)
    c.TickLabels{i};
end
c.Label.String = 'Cluster';
c.Label.FontSize=14;
xlabel('Longitude','FontSize',14);
ylabel('Latitude','FontSize',14);

hold off;

