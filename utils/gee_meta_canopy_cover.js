// DAUIDs to exclude
var excludeList = [];

// Set the projection to Statistics Canada Lambert
var statsCanLambert = ee.Projection('EPSG:3347');

// Access dissemination area shapefiles
var censusSub = ee.FeatureCollection('projects/heroic-glyph-383701/assets/road_buffers_10m_dissemination_area_split')
  .filter(ee.Filter.inList('DAUID', excludeList).not());

// Access Meta 1m Canopy Height Model
var canopyHeight = ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight').mosaic();

// Reproject canopy height to Stats Can Lambert
var canopyHeightReprojected = canopyHeight.reproject({
  crs: statsCanLambert,
  scale: 1
});

// Create binary canopy layer (>= 2m = 1, < 2m = 0)
var canopyBinary = canopyHeightReprojected.gte(2);

// Display the binary canopy layer
Map.addLayer(canopyBinary.selfMask(), {palette: ['green']}, 'Canopy (>=2m)', false);

// Define batch parameters
var totalFeatures = censusSub.size();
var batchSize = 5000; // Process x DAs at a time
var batchNumber = 6;  // Change this for each run (0, 1, 2, 3, etc.)

// Calculate start index
var startIndex = batchNumber * batchSize;

// Get subset of features
var censusSubBatch = ee.FeatureCollection(censusSub.toList(batchSize, startIndex));

print('Total DAs:', totalFeatures);
print('Processing batch:', batchNumber);
print('DAs in this batch:', censusSubBatch.size());

// Function to calculate canopy metrics for each DA
var calculateCanopyMetrics = function(feature) {
  // Explicitly carry DAUID forward to ensure it appears in export
  var dauid = feature.get('DAUID');

  // Get the geometry
  var geometry = feature.geometry();

  // Calculate pixel area in square meters (1m x 1m = 1 sq m per pixel)
  var pixelArea = ee.Image.pixelArea().reproject({
    crs: statsCanLambert,
    scale: 1
  });

  // Calculate total area of the DA in square meters
  var totalArea = pixelArea.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: geometry,
    crs: statsCanLambert,
    scale: 1,
    maxPixels: 1e13
  }).get('area');

  // Calculate canopy area (where binary = 1)
  var canopyAreaImage = canopyBinary.multiply(pixelArea);
  var canopyAreaResult = canopyAreaImage.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: geometry,
    crs: statsCanLambert,
    scale: 1,
    maxPixels: 1e13
  });

  // Get the first (and only) value from the result
  var canopyArea = ee.Number(canopyAreaResult.values().get(0));

  // Convert to square kilometers
  var totalAreaKm2 = ee.Number(totalArea).divide(1e6);
  var canopyAreaKm2 = canopyArea.divide(1e6);

  // Calculate proportion as percentage
  var canopyProportion = canopyAreaKm2.divide(totalAreaKm2).multiply(100);

  // Add properties to feature, explicitly setting DAUID to ensure it's present
  return feature.set({
    'DAUID': dauid,
    'total_area_km2': totalAreaKm2,
    'canopy_area_km2': canopyAreaKm2,
    'canopy_proportion': canopyProportion
  });
};

// Calculate metrics for batch of DAs
var censusWithCanopy = censusSubBatch.map(calculateCanopyMetrics);

// Set map center to Canada and zoom level
Map.setCenter(-96, 62, 4);

// Display the dissemination areas
Map.addLayer(censusSub, {color: 'blue'}, 'All Dissemination Areas', false);
Map.addLayer(censusSubBatch, {color: 'red'}, 'Current Batch', true);

// Print results
print('Dissemination Areas with Canopy Metrics:', censusWithCanopy);
print('Number of DAs in batch:', censusWithCanopy.size());

// Export results to CSV with batch number in filename
Export.table.toDrive({
  collection: censusWithCanopy,
  description: 'canopy_cover_road_network_da_batch_' + batchNumber,
  fileFormat: 'CSV',
  selectors: ['DAUID', 'total_area_km2', 'canopy_area_km2', 'canopy_proportion']
});

print('Export task created. Check the Tasks tab to run it.');