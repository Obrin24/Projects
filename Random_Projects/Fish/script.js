document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("gameCanvas");
    const ctx = canvas.getContext("2d");

    // Fish attributes
    let fish = {
        x: 50,
        y: canvas.height / 2,
        width: 80,
        height: 40,
        speed: 5,
        color: "#ffb347" // Orange color
    };

    // Hoop attributes
    const hoop = {
        x: canvas.width - 150,
        y: canvas.height / 2 - 5,
        width: 100,
        height: 10,
        color: "#ff6b6b" // Red color
    };

    // Bowl attributes
    const bowl = {
        x: 50,
        y: canvas.height / 2 - 100,
        width: canvas.width - 100,
        height: 200,
        color: "#64b5f6" // Blue color
    };

    // Main drawing function
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw bowl
        ctx.fillStyle = bowl.color;
        ctx.fillRect(bowl.x, bowl.y, bowl.width, bowl.height);

        // Draw hoop
        ctx.fillStyle = hoop.color;
        ctx.fillRect(hoop.x, hoop.y, hoop.width, hoop.height);

        // Draw fish body
        ctx.fillStyle = fish.color;
        ctx.beginPath();
        ctx.ellipse(fish.x + fish.width / 2, fish.y + fish.height / 2, fish.width / 2, fish.height / 2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.closePath();

        // Draw fish tail
        ctx.beginPath();
        ctx.moveTo(fish.x + fish.width, fish.y + fish.height / 2);
        ctx.lineTo(fish.x + fish.width + 30, fish.y + fish.height / 2 - 20);
        ctx.lineTo(fish.x + fish.width + 30, fish.y + fish.height / 2 + 20);
        ctx.closePath();
        ctx.fill();

        // Update fish position
        fish.x += fish.speed;

        // Reset fish position if it moves out of canvas
        if (fish.x > canvas.width) {
            fish.x = -fish.width;
        }

        requestAnimationFrame(draw);
    }

    draw();
});
