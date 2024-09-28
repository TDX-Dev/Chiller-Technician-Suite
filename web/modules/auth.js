const root = require('../index')

root.addAPIRegistry('post', '/auth/userlogin', (req, res) => {
    res.send(req.body);
})