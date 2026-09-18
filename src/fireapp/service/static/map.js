// TODO: инициализировать карту, дать пользователю нарисовать AOI (polygon/bbox),
// выбрать интервал дат, дёрнуть /api/hotspots, /api/burns, /api/report и отрисовать результат.

const map = new maplibregl.Map({
  container: "map",
  style: "https://demotiles.maplibre.org/style.json",
  center: [44, 48],
  zoom: 5,
});

map.addControl(new maplibregl.NavigationControl());

async function fetchReport(polygonGeoJson, dateFrom, dateTo) {
  const res = await fetch("/api/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      polygon_geojson: polygonGeoJson,
      date_from: dateFrom,
      date_to: dateTo,
    }),
  });
  return res.json();
}
