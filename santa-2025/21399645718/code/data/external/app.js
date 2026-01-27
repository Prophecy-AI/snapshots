document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const fileSelector = document.getElementById('file-selector');
    const saveButton = document.getElementById('save-button');
    const canvas = document.getElementById('canvas');
    const infoBox = document.getElementById('info-box');

    // --- Constants ---
    const SVG_NS = "http://www.w3.org/2000/svg";
    const TREE_POLYGON_POINTS = "0.0,0.8 0.125,0.5 0.0625,0.5 0.2,0.25 0.1,0.25 0.35,0.0 0.075,0.0 0.075,-0.2 -0.075,-0.2 -0.075,0.0 -0.35,0.0 -0.1,0.25 -0.2,0.25 -0.0625,0.5 -0.125,0.5";
    const basePolygon = TREE_POLYGON_POINTS.split(' ').map(p => {
        const [x, y] = p.split(',').map(Number);
        return { x, y };
    });

    const basePolygonParts = [
        [basePolygon[0], basePolygon[1], basePolygon[14]],
        [basePolygon[14], basePolygon[1], basePolygon[2], basePolygon[13]],
        [basePolygon[13], basePolygon[2], basePolygon[3], basePolygon[12]],
        [basePolygon[12], basePolygon[3], basePolygon[4], basePolygon[11]],
        [basePolygon[11], basePolygon[4], basePolygon[5], basePolygon[10]],
        [basePolygon[10], basePolygon[5], basePolygon[6], basePolygon[9]],
        [basePolygon[9], basePolygon[6], basePolygon[7], basePolygon[8]]
    ];

    // --- State ---
    let trees = [];
    let selectedTreeId = null;
    let originalSideLength = null;
    let isDragging = false;
    let isPanning = false;
    let dragOffset = { x: 0, y: 0 };
    let panStart = { x: 0, y: 0 };
    let viewBox = { x: -25, y: -25, w: 50, h: 50 };

    // --- Viridis Colormap ---
    const viridis = (t) => {
        const colors = [
            [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
            [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
            [180, 222, 44], [253, 231, 37]
        ];
        if (t <= 0) return `rgb(${colors[0][0]}, ${colors[0][1]}, ${colors[0][2]})`;
        if (t >= 1) return `rgb(${colors[colors.length - 1][0]}, ${colors[colors.length - 1][1]}, ${colors[colors.length - 1][2]})`;

        const i = t * (colors.length - 1);
        const i0 = Math.floor(i);
        const i1 = Math.ceil(i);
        const frac = i - i0;

        const c0 = colors[i0];
        const c1 = colors[i1];

        const r = Math.round(c0[0] * (1 - frac) + c1[0] * frac);
        const g = Math.round(c0[1] * (1 - frac) + c1[1] * frac);
        const b = Math.round(c0[2] * (1 - frac) + c1[2] * frac);

        return `rgb(${r}, ${g}, ${b})`;
    };

    // --- Initialization ---
    const init = () => {
        setupEventListeners();
        loadFileList();
    };

    // --- Event Listeners ---
    const setupEventListeners = () => {
        fileSelector.addEventListener('change', (e) => loadTreeData(e.target.value));
        saveButton.addEventListener('click', saveChanges);

        canvas.addEventListener('mousedown', onCanvasMouseDown);
        canvas.addEventListener('wheel', onCanvasWheel);
        
        window.addEventListener('mousemove', onWindowMouseMove);
        window.addEventListener('mouseup', onWindowMouseUp);
        window.addEventListener('keydown', onWindowKeyDown);
    };

    // --- API Calls ---
    const loadFileList = async () => {
        try {
            const response = await fetch('/api/files');
            const files = await response.json();
            
            fileSelector.innerHTML = files.map(f => `<option value="${f}">${f}</option>`).join('');
            
            if (files.length > 0) {
                const defaultFile = files.find(f => f.startsWith('0200_')) || files[0];
                fileSelector.value = defaultFile;
                loadTreeData(defaultFile);
            }
        } catch (error) {
            console.error("Failed to load file list:", error);
        }
    };

    const loadTreeData = async (filename) => {
        try {
            const response = await fetch(`/api/data/${filename}`);
            const data = await response.json();
            trees = data; // Assuming the API returns the array of trees directly
            selectedTreeId = null;
            render();
            resetViewBox();
            const bounds = getOverallBounds();
            const sideLength = bounds ? Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) : null;
            originalSideLength = sideLength;
            updateInfoBox(null, sideLength);
        } catch (error) {
            console.error(`Failed to load data for ${filename}:`, error);
        }
    };

    const saveChanges = async () => {
        const filename = fileSelector.value;
        if (!filename) {
            alert('No file selected.');
            return;
        }
        try {
            const response = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, trees })
            });
            if (response.ok) {
                alert('Changes saved successfully!');
            } else {
                const error = await response.text();
                alert(`Failed to save changes: ${error}`);
            }
        } catch (error) {
            console.error("Failed to save changes:", error);
        }
    };

    // --- Collision Detection ---


    const getRotatedVertices = (tree, polygon) => {
        const angleRad = tree.angle * Math.PI / 180;
        const cos = Math.cos(angleRad);
        const sin = Math.sin(angleRad);

        return polygon.map(p => ({
            x: p.x * cos - p.y * sin + tree.x,
            y: p.x * sin + p.y * cos + tree.y
        }));
    };

    const project = (vertices, axis) => {
        let min = Infinity;
        let max = -Infinity;
        for (const v of vertices) {
            const dot = v.x * axis.x + v.y * axis.y;
            min = Math.min(min, dot);
            max = Math.max(max, dot);
        }
        return { min, max };
    };

    const overlap = (p1, p2) => {
        return p1.max > p2.min && p2.max > p1.min;
    };

    const getAxes = (vertices) => {
        const axes = [];
        for (let i = 0; i < vertices.length; i++) {
            const p1 = vertices[i];
            const p2 = vertices[i + 1 === vertices.length ? 0 : i + 1];
            const edge = { x: p1.x - p2.x, y: p1.y - p2.y };
            const normal = { x: -edge.y, y: edge.x };
            const length = Math.sqrt(normal.x * normal.x + normal.y * normal.y);
            if (length > 0) {
                normal.x /= length;
                normal.y /= length;
                axes.push(normal);
            }
        }
        return axes;
    };

    const checkPolygonCollision = (vertices1, vertices2) => {
        const axes1 = getAxes(vertices1);
        const axes2 = getAxes(vertices2);

        for (const axis of axes1) {
            const p1 = project(vertices1, axis);
            const p2 = project(vertices2, axis);
            if (!overlap(p1, p2)) return false;
        }

        for (const axis of axes2) {
            const p1 = project(vertices1, axis);
            const p2 = project(vertices2, axis);
            if (!overlap(p1, p2)) return false;
        }

        return true;
    };

    const checkTreeCollision = (tree1, tree2) => {
        // AABB check first (broad phase)
        const getBounds = (tree) => {
            const angleRad = tree.angle * Math.PI / 180;
            const cos = Math.cos(angleRad);
            const sin = Math.sin(angleRad);
            const rotatedPoints = basePolygon.map(p => ({
                x: p.x * cos - p.y * sin,
                y: p.x * sin + p.y * cos
            }));
            const xs = rotatedPoints.map(p => p.x);
            const ys = rotatedPoints.map(p => p.y);
            return {
                minX: Math.min(...xs) + tree.x,
                minY: Math.min(...ys) + tree.y,
                maxX: Math.max(...xs) + tree.x,
                maxY: Math.max(...ys) + tree.y,
            };
        };
        const bounds1 = getBounds(tree1);
        const bounds2 = getBounds(tree2);
        const aabbCollision = bounds1.minX < bounds2.maxX && bounds1.maxX > bounds2.minX &&
                              bounds1.minY < bounds2.maxY && bounds1.maxY > bounds2.minY;

        if (!aabbCollision) return false;

        // SAT check (narrow phase)
        for (const part1 of basePolygonParts) {
            const vertices1 = getRotatedVertices(tree1, part1);
            for (const part2 of basePolygonParts) {
                const vertices2 = getRotatedVertices(tree2, part2);
                if (checkPolygonCollision(vertices1, vertices2)) {
                    return true;
                }
            }
        }

        return false;
    };

    // --- Rendering ---
    const render = () => {
        canvas.innerHTML = '';
        if (trees.length === 0) {
            updateInfoBox(null, null);
            return;
        }

        // --- Collision Detection ---
        const collisions = new Set();
        for (let i = 0; i < trees.length; i++) {
            for (let j = i + 1; j < trees.length; j++) {
                if (checkTreeCollision(trees[i], trees[j])) {
                    collisions.add(trees[i].id);
                    collisions.add(trees[j].id);
                }
            }
        }

        // --- Bounding Box and Edge Trees ---
        const touchingTrees = new Set();
        const allBounds = trees.map(tree => ({ id: tree.id, ...getRotatedPolygonBounds(tree) }));
        
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        allBounds.forEach(b => {
            minX = Math.min(minX, b.minX);
            maxX = Math.max(maxX, b.maxX);
            minY = Math.min(minY, b.minY);
            maxY = Math.max(maxY, b.maxY);
        });
        const overallBounds = { minX, minY, maxX, maxY };

        const precision = 1e6;
        const roundedMinX = Math.round(minX * precision);
        const roundedMaxX = Math.round(maxX * precision);
        const roundedMinY = Math.round(minY * precision);
        const roundedMaxY = Math.round(maxY * precision);

        allBounds.forEach(b => {
            if (
                Math.round(b.minX * precision) === roundedMinX ||
                Math.round(b.maxX * precision) === roundedMaxX ||
                Math.round(b.minY * precision) === roundedMinY ||
                Math.round(b.maxY * precision) === roundedMaxY
            ) {
                touchingTrees.add(b.id);
            }
        });

        // --- SVG Rendering ---
        canvas.setAttribute('transform', 'scale(1, -1)');
        const colors = trees.map((_, i) => viridis(i / (trees.length - 1 || 1)));

        trees.forEach((tree, i) => {
            const group = document.createElementNS(SVG_NS, 'g');
            group.setAttribute('transform', `translate(${tree.x} ${tree.y}) rotate(${tree.angle})`);
            group.dataset.id = tree.id;
            if (tree.id === selectedTreeId) {
                group.classList.add('selected');
            }
            if (collisions.has(tree.id)) {
                group.classList.add('colliding');
            }

            const polygon = document.createElementNS(SVG_NS, 'polygon');
            polygon.setAttribute('points', TREE_POLYGON_POINTS);
            polygon.setAttribute('class', 'tree-polygon');
            
            if (touchingTrees.has(tree.id)) {
                polygon.setAttribute('fill', 'black');
            } else {
                polygon.setAttribute('fill', colors[i]);
            }
            
            polygon.setAttribute('fill-opacity', '0.5');
            
            group.appendChild(polygon);
            canvas.appendChild(group);

            group.addEventListener('mousedown', (e) => onTreeMouseDown(e, tree.id));
        });

        renderBoundingSquare(overallBounds);
        
        const sideLength = Math.max(overallBounds.maxX - overallBounds.minX, overallBounds.maxY - overallBounds.minY);
        updateInfoBox(selectedTreeId ? trees.find(t => t.id === selectedTreeId) : null, sideLength);
    };

    const renderBoundingSquare = (overallBounds) => {
        if (!overallBounds) return;

        const { minX, minY, maxX, maxY } = overallBounds;
        const width = maxX - minX;
        const height = maxY - minY;

        const sideLength = Math.max(width, height);
        const squareX = width >= height ? minX : minX - (sideLength - width) / 2;
        const squareY = height > width ? minY : minY - (sideLength - height) / 2;

        const boundingBox = document.createElementNS(SVG_NS, 'rect');
        boundingBox.setAttribute('x', squareX);
        boundingBox.setAttribute('y', squareY);
        boundingBox.setAttribute('width', sideLength);
        boundingBox.setAttribute('height', sideLength);
        boundingBox.setAttribute('class', 'bounding-box');
        
        canvas.appendChild(boundingBox);
    };

    // --- Interaction Handlers ---
    const onCanvasMouseDown = (e) => {
        isPanning = true;
        panStart = { x: e.clientX, y: e.clientY };
    };

    const onTreeMouseDown = (e, treeId) => {
        e.stopPropagation();
        isDragging = true;
        selectedTreeId = treeId;
        
        const tree = trees.find(t => t.id === treeId);
        const CTM = canvas.getScreenCTM().inverse();
        const mousePos = {
            x: e.clientX * CTM.a + CTM.e,
            y: e.clientY * CTM.d + CTM.f
        };
        
        dragOffset.x = mousePos.x - tree.x;
        dragOffset.y = mousePos.y - tree.y;

        render(); // Re-render to show selection and update info box
    };

    const onWindowMouseMove = (e) => {
        const CTM = canvas.getScreenCTM().inverse();
        const mousePos = {
            x: e.clientX * CTM.a + CTM.e,
            y: e.clientY * CTM.d + CTM.f
        };

        if (isDragging && selectedTreeId !== null) {
            const tree = trees.find(t => t.id === selectedTreeId);
            if (tree) {
                tree.x = mousePos.x - dragOffset.x;
                tree.y = mousePos.y - dragOffset.y;
                render();
            }
        } else if (isPanning) {
            const dx = (e.clientX - panStart.x) * (viewBox.w / canvas.clientWidth);
            const dy = (e.clientY - panStart.y) * (viewBox.h / canvas.clientHeight);
            viewBox.x -= dx;
            viewBox.y += dy; // Y is inverted
            updateViewBox();
            panStart = { x: e.clientX, y: e.clientY };
        }
    };

    const onWindowMouseUp = () => {
        isDragging = false;
        isPanning = false;
    };

    const onCanvasWheel = (e) => {
        e.preventDefault();
        const CTM = canvas.getScreenCTM().inverse();
        const mousePos = {
            x: e.clientX * CTM.a + CTM.e,
            y: e.clientY * CTM.d + CTM.f
        };
        
        const zoomFactor = e.deltaY < 0 ? 0.9 : 1.1;
        const newW = viewBox.w * zoomFactor;
        const newH = viewBox.h * zoomFactor;

        viewBox.x = mousePos.x - (mousePos.x - viewBox.x) * zoomFactor;
        viewBox.y = mousePos.y - (mousePos.y - viewBox.y) * zoomFactor;
        viewBox.w = newW;
        viewBox.h = newH;
        
        updateViewBox();
    };

    const onWindowKeyDown = (e) => {
        if (selectedTreeId === null) return;
        const tree = trees.find(t => t.id === selectedTreeId);
        if (!tree) return;

        if (e.key === 'r' || e.key === 'R') {
            const increment = e.shiftKey ? -5 : 5;
            tree.angle = (tree.angle + increment + 360) % 360;
            render();
        }
    };

    // --- UI Update ---
    const updateInfoBox = (tree, sideLength = null) => {
        const improvement = sideLength !== null && originalSideLength !== null ? sideLength - originalSideLength : 0;
        const improvementText = improvement.toFixed(8);
        const improvementColor = improvement > 0 ? 'red' : 'green';

        if (!tree) {
            infoBox.innerHTML = `
                <strong>No tree selected</strong><br>
                Bounding Box Side: ${sideLength !== null ? sideLength.toFixed(8) : 'N/A'}<br>
                Improvement: <span style="color: ${improvementColor};">${improvementText}</span><br>
            `;
            return;
        }
        infoBox.innerHTML = `
            <strong>Selected Tree</strong><br>
            ID: ${tree.id}<br>
            X: <input type="number" id="tree-x" value="${tree.x.toFixed(8)}" step="0.001"><br>
            Y: <input type="number" id="tree-y" value="${tree.y.toFixed(8)}" step="0.001"><br>
            Angle: <input type="number" id="tree-angle" value="${tree.angle.toFixed(2)}" step="1"><br>
            Bounding Box Side: ${sideLength !== null ? sideLength.toFixed(8) : 'N/A'}<br>
            Improvement: <span style="color: ${improvementColor};">${improvementText}</span><br>
        `;
        infoBox.querySelector('#tree-x').addEventListener('change', (e) => updateTreeFromInput('x', e.target.value));
        infoBox.querySelector('#tree-y').addEventListener('change', (e) => updateTreeFromInput('y', e.target.value));
        infoBox.querySelector('#tree-angle').addEventListener('change', (e) => updateTreeFromInput('angle', e.target.value));
    };

    const updateTreeFromInput = (prop, value) => {
        if (selectedTreeId === null) return;
        const tree = trees.find(t => t.id === selectedTreeId);
        if (tree) {
            tree[prop] = parseFloat(value);
            render();
        }
    };

    // --- Geometry & ViewBox ---
    const getRotatedPolygonBounds = (tree) => {
        const angleRad = tree.angle * Math.PI / 180;
        const cos = Math.cos(angleRad);
        const sin = Math.sin(angleRad);

        const rotatedPoints = basePolygon.map(p => ({
            x: p.x * cos - p.y * sin,
            y: p.x * sin + p.y * cos
        }));

        const xs = rotatedPoints.map(p => p.x);
        const ys = rotatedPoints.map(p => p.y);
        
        return {
            minX: Math.min(...xs) + tree.x,
            minY: Math.min(...ys) + tree.y,
            maxX: Math.max(...xs) + tree.x,
            maxY: Math.max(...ys) + tree.y,
        };
    };

    const getOverallBounds = () => {
        if (trees.length === 0) return null;
        
        const allBounds = trees.map(getRotatedPolygonBounds);
        
        return {
            minX: Math.min(...allBounds.map(b => b.minX)),
            minY: Math.min(...allBounds.map(b => b.minY)),
            maxX: Math.max(...allBounds.map(b => b.maxX)),
            maxY: Math.max(...allBounds.map(b => b.maxY)),
        };
    };

    const updateViewBox = () => {
        canvas.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
    };

    const resetViewBox = () => {
        const bounds = getOverallBounds();
        if (!bounds) {
            viewBox = { x: -25, y: -25, w: 50, h: 50 };
        } else {
            const padding = 2;
            const width = bounds.maxX - bounds.minX;
            const height = bounds.maxY - bounds.minY;
            const size = Math.max(width, height) + padding * 2;
            
            const centerX = (bounds.minX + bounds.maxX) / 2;
            const centerY = (bounds.minY + bounds.maxY) / 2;

            viewBox.w = size;
            viewBox.h = size;
            viewBox.x = centerX - size / 2;
            viewBox.y = -(centerY + size / 2); // Invert Y for SVG
        }
        updateViewBox();
    };

    // --- Start the application ---
    init();
});
