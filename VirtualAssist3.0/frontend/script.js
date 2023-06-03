document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    const input = document.querySelector("input[name='message']");
    const canvas = document.querySelector(".canvas");

    form.addEventListener("submit", function(event) {
        event.preventDefault();

        const message = input.value.trim();

        if (message !== "") {
            const newMessage = createMessage(message, "user-message");
            canvas.appendChild(newMessage);
            input.value = "";
            input.focus();
        }
    });

    function createMessage(content, className) {
        const messageContainer = document.createElement("div");
        messageContainer.classList.add("message", className);

        const messageContent = document.createElement("p");
        messageContent.classList.add("message-content");
        messageContent.textContent = content;

        messageContainer.appendChild(messageContent);

        return messageContainer;
    }
});
