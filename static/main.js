const img = document.getElementById("img")

function file_img() {
    let fileInput = event.target
    var file = fileInput.files[0]

    return file
}

function setUp(event) {
    if (file_img()) {
        const reader = new FileReader()

        reader.onload = function(event) {
            img.style.display = "block"
            img.src = event.target.result
        }

        reader.readAsDataURL(file_img())
    }
}