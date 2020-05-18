#include <signal.h> 
#include <winsock2.h>
#include <ws2tcpip.h>
#include <memory>
#include <cstdint>
#include <ostream>
#include <evhttp.h>
#include <string>

class HttpServer {
public:

    struct WSAContext {
        WSAContext(int major_verion, int minor_version);
        ~WSAContext();
        WSADATA wsaData_;
    };

    //struct EventBaseContext {
    //    EventBaseContext();
    //    EventBaseContext(event_config *config);
    //    ~EventBaseContext();
    //    event_base *base;
    //};

    //struct EventHTTPContext {
    //    EventHTTPContext();
    //    EventHTTPContext(EventBaseContext &base);
    //    ~EventHTTPContext();

    //    EventHTTPContext(const EventHTTPContext &cpy) = delete;
    //    EventHTTPContext & operator = (const EventHTTPContext &cpy) = delete;

    //    EventHTTPContext(EventHTTPContext &&cpy);
    //    EventHTTPContext & operator = (EventHTTPContext &&cpy);

    //    evhttp *http;
    //};

    HttpServer(const std::string &host, std::uint16_t port, std::ostream & output);
    ~HttpServer();

    HttpServer(const HttpServer &cpy) = delete;
    HttpServer & operator = (const HttpServer &cpy) = delete;

    const std::string &getURL() const;

    //INFO: NEEDS TO BIND with C API
    void default_handler(struct evhttp_request *req, void *arg);
    void default_signal_handler(evutil_socket_t sig, short events, void *arg);
private:
     
     bool display_listen_socket();
private:
    std::string host_;
    std::uint16_t port_;
    std::ostream &output_;

    WSAContext wsa_context_;
    //EventBaseContext base_;
    event_base *base_;
    //EventHTTPContext http_;
    evhttp *http_;
    evhttp_bound_socket *handle_;
    struct event *termination_event_;

    std::string url_;
};