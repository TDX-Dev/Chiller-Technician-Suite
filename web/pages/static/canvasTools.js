const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


function addCanvas(id, contId, name) {
    const mainContainer = document.getElementById(contId);

    const mainWidgetContainer = document.createElement('div');
    mainWidgetContainer.setAttribute('class', 'main-widget-container');

    const canvas = document.createElement('canvas');
    canvas.setAttribute('id', id);
    canvas.setAttribute('height', '200')

    mainWidgetContainer.innerText += name;
    mainWidgetContainer.appendChild(canvas);
    mainContainer.appendChild(mainWidgetContainer);

    return canvas.getContext('2d')
}