const submitButton = document.getElementById('submitButton');
submitButton.onclick = userSubmitEventHandler;


function userSubmitEventHandler(e) {
    let query = {}
    e.preventDefault()
    query = {
        query: "GOOG"
    }
    fetch('http://127.0.0.1:5000/', {
        method: 'POST',
        body: JSON.stringify(query),
        headers: {
            'Content-type': 'application/json; charset=UTF-8'
        }
    })
    .then(response => response.json())
    .then(json => {
        console.log(json)
       document.getElementById("demo").innerHTML = json;
    })
}