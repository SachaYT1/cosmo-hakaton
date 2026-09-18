"use strict";

const SEVERITY_COLORS = { 1: "#ffd23f", 2: "#ff8c1a", 3: "#d7191c" };
const SEVERITY_LABELS = { 1: "слабая", 2: "средняя", 3: "сильная" };

const map = L.map("map", { preferCanvas: true }).setView([48.0, 44.0], 6);
map.attributionControl.setPrefix(false);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const drawnItems = new L.FeatureGroup().addTo(map);
const contoursLayer = L.layerGroup().addTo(map);
const hotspotsLayer = L.layerGroup().addTo(map);
let demo = null;

new L.Control.Draw({
  draw: {
    polygon: { allowIntersection: false, showArea: true },
    rectangle: {},
    polyline: false, circle: false, marker: false, circlemarker: false,
  },
  edit: { featureGroup: drawnItems, edit: false },
}).addTo(map);

map.on(L.Draw.Event.CREATED, (e) => {
  drawnItems.clearLayers();
  drawnItems.addLayer(e.layer);
});

// Граница территории мониторинга (пунктир)
fetch("aoi.geojson")
  .then((r) => (r.ok ? r.json() : null))
  .then((geo) => {
    if (!geo) return;
    const aoi = (geo.features || []).filter((f) => f.id === "aoi");
    L.geoJSON(aoi.length ? { type: "FeatureCollection", features: aoi } : geo, {
      style: { color: "#666", weight: 1.5, dashArray: "6 4", fill: false },
    }).addTo(map);
  })
  .catch(() => {});

// Метаданные каталога: диапазон дат по умолчанию + демо-пример
fetch("/api/meta")
  .then((r) => r.json())
  .then((m) => {
    document.getElementById("date-from").value = m.date_min;
    document.getElementById("date-to").value = m.date_max;
    demo = m.demo;
  })
  .catch(() => {});

function currentGeometry() {
  const layers = drawnItems.getLayers();
  return layers.length ? layers[0].toGeoJSON().geometry : null;
}

function requestBody(geometry) {
  return {
    polygon: geometry,
    date_from: document.getElementById("date-from").value,
    date_to: document.getElementById("date-to").value,
  };
}

function showError(msg) {
  const el = document.getElementById("error");
  el.textContent = msg;
  el.classList.remove("hidden");
}

function clearError() {
  document.getElementById("error").classList.add("hidden");
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(typeof d.detail === "string" ? d.detail : `Ошибка API (HTTP ${r.status})`);
  }
  return r.json();
}

function renderHotspots(fc) {
  hotspotsLayer.clearLayers();
  L.geoJSON(fc, {
    pointToLayer: (_f, latlng) =>
      L.circleMarker(latlng, { radius: 4, color: "#7a0000", weight: 1, fillColor: "#ff3b30", fillOpacity: 0.9 }),
    onEachFeature: (f, layer) =>
      layer.bindPopup(
        `<b>Термоточка</b><br>Съёмка: ${f.properties.acq_datetime}<br>Спутник: ${f.properties.satellite}`
      ),
  }).addTo(hotspotsLayer);
}

function renderContours(fc) {
  contoursLayer.clearLayers();
  L.geoJSON(fc, {
    style: (f) => ({
      color: SEVERITY_COLORS[f.properties.severity] || "#999",
      weight: 1,
      fillColor: SEVERITY_COLORS[f.properties.severity] || "#999",
      fillOpacity: 0.45,
    }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      layer.bindPopup(
        `<b>Гарь ${p.contour_id}</b><br>Степень: ${p.severity} (${SEVERITY_LABELS[p.severity]})` +
          `<br>Площадь: ${p.area_ha} га<br>Период: ${p.date_pre} — ${p.date_post}`
      );
    },
  }).addTo(contoursLayer);
}

function renderReport(rep) {
  const rows = rep.by_severity
    .map(
      (r) =>
        `<tr><td><span class="swatch" style="background:${SEVERITY_COLORS[r.severity]}"></span></td>` +
        `<td>${r.severity} — ${r.label}</td><td>${r.area_ha.toFixed(2)} га</td><td>${r.share_pct}%</td></tr>`
    )
    .join("");
  document.getElementById("report-body").innerHTML =
    `<table><tr><th></th><th>Степень</th><th>Площадь</th><th>Доля</th></tr>${rows}` +
    `<tr class="total-row"><td></td><td>Итого гарей</td><td>${rep.total_burned_area_ha.toFixed(2)} га</td><td></td></tr>` +
    `<tr><td></td><td>Термоточек</td><td colspan="2">${rep.hotspot_count}</td></tr>` +
    `<tr><td></td><td>Контуров</td><td colspan="2">${rep.contour_count}</td></tr></table>`;
  document.getElementById("report").classList.remove("hidden");
}

// Порог отрисовки мелких контуров в зависимости от размера области запроса.
// Только для карты: справка и выгрузки всегда считаются по всем контурам.
function mapMinAreaHa() {
  const layers = drawnItems.getLayers();
  if (!layers.length) return 0;
  const b = layers[0].getBounds();
  const midLat = ((b.getSouth() + b.getNorth()) / 2) * (Math.PI / 180);
  const areaKm2 =
    (b.getEast() - b.getWest()) * 111.32 * Math.cos(midLat) *
    (b.getNorth() - b.getSouth()) * 110.57;
  if (areaKm2 > 50000) return 5;
  if (areaKm2 > 5000) return 1;
  if (areaKm2 > 500) return 0.2;
  return 0;
}

function renderFilterNote(minArea) {
  const el = document.getElementById("filter-note");
  if (minArea > 0) {
    el.textContent = `Для скорости отрисовки на карте скрыты контуры мельче ${minArea} га. ` +
      "Справка и выгрузки учитывают все контуры.";
    el.classList.remove("hidden");
  } else {
    el.classList.add("hidden");
  }
}

async function runQuery() {
  clearError();
  const geometry = currentGeometry();
  if (!geometry) {
    showError("Сначала нарисуйте полигон или прямоугольник на карте");
    return;
  }
  const body = requestBody(geometry);
  const minArea = mapMinAreaHa();
  const btn = document.getElementById("btn-query");
  btn.disabled = true;
  try {
    const [hs, ba, rep] = await Promise.all([
      apiPost("/api/hotspots", body),
      apiPost("/api/burned-areas", { ...body, min_area_ha: minArea }),
      apiPost("/api/report", body),
    ]);
    renderHotspots(hs);
    renderContours(ba);
    renderReport(rep);
    renderFilterNote(minArea);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("btn-query").addEventListener("click", runQuery);

document.getElementById("btn-demo").addEventListener("click", () => {
  if (!demo) {
    showError("Демо-пример недоступен: каталог пуст");
    return;
  }
  const [w, s, e, n] = demo.bbox;
  drawnItems.clearLayers();
  drawnItems.addLayer(L.rectangle([[s, w], [n, e]], { color: "#8ab4f8", weight: 2, fill: false }));
  document.getElementById("date-from").value = demo.date_from;
  document.getElementById("date-to").value = demo.date_to;
  map.fitBounds([[s, w], [n, e]], { padding: [30, 30] });
  runQuery();
});

document.querySelectorAll("[data-export]").forEach((btn) =>
  btn.addEventListener("click", async () => {
    clearError();
    const geometry = currentGeometry();
    if (!geometry) {
      showError("Нарисуйте область для выгрузки");
      return;
    }
    const name = btn.dataset.export;
    const r = await fetch(`/api/export/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody(geometry)),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      showError(typeof d.detail === "string" ? d.detail : `Ошибка выгрузки (HTTP ${r.status})`);
      return;
    }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name.replace(".shp.zip", "_shp.zip");
    a.click();
    URL.revokeObjectURL(a.href);
  })
);
